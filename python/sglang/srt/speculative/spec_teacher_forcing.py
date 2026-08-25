"""Teacher-forcing measurement mode for speculative decoding.

Gated by ``SGLANG_SPEC_TEACHER_FORCING`` plus a per-request
``sampling_params.custom_params["spec_teacher_forcing_ids"]`` carrying the
ground-truth output token ids of a prior *base* (non-speculative) run.

When active, every verify step is scored against the base trajectory instead of
the target model's own argmax, and then **all draft tokens are rejected** so the
step commits exactly one token -- the next base token. The accept length that
*would* have been reached is recorded per step and surfaced as
``meta_info["spec_teacher_forcing_accept_length"]`` (bonus token included, one
entry per output token, ``None`` for the prefill token).

Because every step advances by exactly one token, each output position is
measured under the same ground-truth prefix, and EAGLE3 / DFLASH / NGRAM are all
measured on an identical trajectory.

Known measurement artifact: near the end of the base sequence the comparison
window runs out of real tokens and is padded with a sentinel that never matches,
so the last ``draft_token_num`` steps are truncated (biased low). The sentinel is
confined to the verify kernels -- see ``force_reject_all_drafts_tree``.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import torch

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.layers.logits_processor import LogitsProcessorOutput
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

# Key inside sampling_params.custom_params carrying the base run's output ids.
SPEC_TEACHER_FORCING_IDS_KEY = "spec_teacher_forcing_ids"
# meta_info key carrying the per-step accept length (bonus included).
SPEC_TEACHER_FORCING_ACCEPT_LENGTH_KEY = "spec_teacher_forcing_accept_length"

SPEC_TEACHER_FORCING_ENABLED = envs.SGLANG_SPEC_TEACHER_FORCING.get()

# Only these three workers carry the matching prefill-token override; any other
# algorithm would get verify-side forcing without it and silently mismeasure.
SUPPORTED_ALGORITHMS = ("EAGLE3", "DFLASH", "NGRAM")

# Fills the comparison window past the end of the base sequence. Draft token ids
# are always non-negative, so it can never match and the accept walk stops at the
# real data boundary.
PAD_TOKEN = -1


def validate_server_args(server_args: ServerArgs) -> None:
    """Reject a server configuration that cannot produce a clean measurement."""
    if not SPEC_TEACHER_FORCING_ENABLED:
        return

    algorithm = (server_args.speculative_algorithm or "").upper()
    if algorithm not in SUPPORTED_ALGORITHMS:
        raise ValueError(
            "SGLANG_SPEC_TEACHER_FORCING requires --speculative-algorithm to be "
            f"one of {SUPPORTED_ALGORITHMS}, got {server_args.speculative_algorithm!r}."
        )

    required = {
        "max_running_requests": (server_args.max_running_requests, 1),
        "chunked_prefill_size": (server_args.chunked_prefill_size, -1),
        "disable_radix_cache": (server_args.disable_radix_cache, True),
        "disable_overlap_schedule": (server_args.disable_overlap_schedule, True),
    }
    wrong = {
        name: actual for name, (actual, want) in required.items() if actual != want
    }
    if wrong:
        raise ValueError(
            "SGLANG_SPEC_TEACHER_FORCING requires "
            "--max-running-requests 1 --chunked-prefill-size -1 "
            "--disable-radix-cache --disable-overlap-schedule; got "
            + ", ".join(f"{name}={value!r}" for name, value in wrong.items())
            + "."
        )

    if envs.SGLANG_SIMULATE_ACC_LEN.get() > 0:
        raise ValueError(
            "SGLANG_SPEC_TEACHER_FORCING and SGLANG_SIMULATE_ACC_LEN both override "
            "acceptance and cannot be enabled together."
        )


def validate_and_clamp_sampling_params(sampling_params: Optional[dict]) -> None:
    """Validate a raw (not yet normalized) /generate sampling_params dict.

    No-op unless the request actually carries base ids, so an unannotated request
    keeps running the normal speculative path.

    Must run before ``SamplingParams`` normalization: ``__post_init__`` rewrites
    ``temperature=0`` into ``temperature=1.0, top_k=1``, so the raw dict is the
    only place the caller's intent is still visible.
    """
    if not SPEC_TEACHER_FORCING_ENABLED or not isinstance(sampling_params, dict):
        return

    custom_params = sampling_params.get("custom_params")
    if not isinstance(custom_params, dict):
        return
    base_output_ids = custom_params.get(SPEC_TEACHER_FORCING_IDS_KEY)
    if base_output_ids is None:
        return

    if not isinstance(base_output_ids, list) or not base_output_ids:
        raise ValueError(
            f"custom_params[{SPEC_TEACHER_FORCING_IDS_KEY!r}] must be a non-empty "
            f"list of token ids, got {type(base_output_ids).__name__}."
        )
    if not all(isinstance(token, int) for token in base_output_ids):
        raise ValueError(
            f"custom_params[{SPEC_TEACHER_FORCING_IDS_KEY!r}] must contain only "
            "integer token ids."
        )

    if sampling_params.get("temperature") != 0:
        raise ValueError(
            "Speculative teacher forcing requires sampling_params temperature=0, "
            f"got {sampling_params.get('temperature')!r}."
        )

    max_new_tokens = sampling_params.get("max_new_tokens")
    if max_new_tokens is None or max_new_tokens > len(base_output_ids):
        sampling_params["max_new_tokens"] = len(base_output_ids)
        logger.info(
            "Speculative teacher forcing: capping max_new_tokens from %r to %d "
            "(length of the base output ids).",
            max_new_tokens,
            len(base_output_ids),
        )


def read_base_output_ids(reqs: List[Req]) -> Optional[List[List[int]]]:
    """Base output ids for every request, or None to skip teacher forcing.

    Returns None when the mode is off or when *any* request lacks base ids: the
    batch then runs the untouched speculative path rather than being partially
    forced.
    """
    if not SPEC_TEACHER_FORCING_ENABLED or not reqs:
        return None

    base_output_ids = []
    for req in reqs:
        custom_params = req.sampling_params.custom_params
        if not isinstance(custom_params, dict):
            return None
        ids = custom_params.get(SPEC_TEACHER_FORCING_IDS_KEY)
        if ids is None:
            return None
        base_output_ids.append(ids)
    return base_output_ids


def build_base_windows(
    *,
    base_output_ids: List[List[int]],
    num_output_tokens: List[int],
    width: int,
    device: str,
) -> torch.Tensor:
    """`[bs, width]` ground-truth continuation of each request.

    Row ``b`` is ``base_output_ids[b][L : L + width]`` where ``L`` is how many
    tokens the request has already emitted, right-padded with ``PAD_TOKEN``.
    Column ``d`` therefore holds the token that must follow a node at depth ``d``.
    """
    rows = []
    for ids, offset in zip(base_output_ids, num_output_tokens):
        window = list(ids[offset : offset + width])
        rows.append(window + [PAD_TOKEN] * (width - len(window)))
    return torch.tensor(rows, dtype=torch.int64, device=device)


def build_forced_target_predicts_tree(
    *,
    windows: torch.Tensor,
    positions: torch.Tensor,
    bs: int,
    draft_token_num: int,
) -> torch.Tensor:
    """`[bs, draft_token_num]` ground-truth prediction for every tree node.

    ``positions`` is absolute and node 0 of each request is the tree root, so the
    node depth is ``positions - positions[:, :1]``. Nodes off the base trajectory
    get harmless values: the greedy walk starts at the root and only descends
    into children whose token equals the base token, so it never reaches them.
    """
    positions_2d = positions.view(bs, draft_token_num)
    depths = (positions_2d - positions_2d[:, :1]).to(torch.int64)
    return torch.gather(windows, 1, depths)


def build_forced_target_predicts_chain(
    *,
    base_output_ids: List[List[int]],
    num_output_tokens: List[int],
    block_size: int,
    device: str,
) -> torch.Tensor:
    """`[bs, block_size]` ground-truth prediction for a flat draft chain.

    Chain position ``t`` predicts ``base_output_ids[L + t]``, which is exactly the
    comparison window -- no depth indirection needed.
    """
    return build_base_windows(
        base_output_ids=base_output_ids,
        num_output_tokens=num_output_tokens,
        width=block_size,
        device=device,
    )


def force_reject_all_drafts_tree(
    *,
    predicts: torch.Tensor,
    accept_indices: torch.Tensor,
    num_correct_drafts: torch.Tensor,
    target_predicts: torch.Tensor,
) -> None:
    """Commit only the bonus token (the next base token); reject every draft.

    Also scrubs ``PAD_TOKEN`` out of ``predicts``: the verify kernel writes the
    comparison value into *every* node, and EAGLE hands the whole ``predicts``
    buffer to the draft extend as ``batch.input_ids``, where a negative id would
    index the embedding table out of bounds. Only rejected nodes are touched, so
    the committed token is unaffected.
    """
    roots = accept_indices[:, 0].to(torch.int64)
    predicts[roots] = target_predicts[:, 0].to(dtype=predicts.dtype)
    predicts.clamp_(min=0)
    accept_indices[:, 1:] = -1
    num_correct_drafts.fill_(0)


def force_reject_all_drafts_chain(
    *,
    num_correct_drafts: torch.Tensor,
    commit_lens: torch.Tensor,
    bonus_tokens: torch.Tensor,
    out_tokens: torch.Tensor,
    target_predicts: torch.Tensor,
) -> None:
    """Chain (DFlash) counterpart of :func:`force_reject_all_drafts_tree`.

    ``out_tokens`` needs no sentinel scrub: positions 1.. hold real draft tokens
    and only position 0 is ever read back (``commit_lens == 1``).
    """
    bonus_tokens.copy_(target_predicts[:, 0].to(dtype=bonus_tokens.dtype))
    num_correct_drafts.fill_(0)
    commit_lens.fill_(1)
    out_tokens[:, 0] = bonus_tokens.to(dtype=out_tokens.dtype)


def record_accept_lengths(
    *,
    logits_output: LogitsProcessorOutput,
    num_accept_tokens: List[int],
) -> None:
    """Stage one accept length per request for this verify step.

    Rides the existing ``customized_info`` pipeline, which appends one element
    per request per step and ends up as ``meta_info[...]``.
    """
    _stage_customized_info(logits_output=logits_output, values=num_accept_tokens)


def apply_prefill_teacher_forcing(
    *,
    reqs: List[Req],
    next_token_ids: torch.Tensor,
    logits_output: LogitsProcessorOutput,
) -> None:
    """Replace the prefill token with the base token and reserve its record slot.

    The prefill token is overridden in place so the draft model's first proposal
    is conditioned on the base trajectory too. The record slot is filled with
    ``None`` because "accept length" is meaningless for prefill -- but it must be
    filled, since the result processor appends one element per request for the
    prefill step as well.
    """
    base_output_ids = read_base_output_ids(reqs)
    if base_output_ids is None:
        return

    forced = [ids[len(req.output_ids)] for ids, req in zip(base_output_ids, reqs)]
    next_token_ids.copy_(
        torch.tensor(forced, dtype=next_token_ids.dtype, device=next_token_ids.device)
    )
    _stage_customized_info(logits_output=logits_output, values=[None] * len(reqs))


def _stage_customized_info(
    *,
    logits_output: LogitsProcessorOutput,
    values: List[Any],
) -> None:
    customized_info: Optional[Dict[str, List[Any]]] = logits_output.customized_info
    if customized_info is None:
        customized_info = {}
        logits_output.customized_info = customized_info
    customized_info[SPEC_TEACHER_FORCING_ACCEPT_LENGTH_KEY] = values

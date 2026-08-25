"""Unit tests for the speculative teacher-forcing measurement mode.

Pure tensor / dict level -- no model, no server, no GPU.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.environ import envs
from sglang.srt.speculative import spec_teacher_forcing as stf
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _make_req(custom_params, num_output_tokens=0):
    return SimpleNamespace(
        sampling_params=SimpleNamespace(custom_params=custom_params),
        output_ids=[0] * num_output_tokens,
    )


class TestBuildBaseWindows(CustomTestCase):
    def test_window_slides_with_emitted_tokens(self):
        windows = stf.build_base_windows(
            base_output_ids=[[100, 101, 102, 103, 104]],
            num_output_tokens=[1],
            width=3,
            device="cpu",
        )
        self.assertEqual(windows.tolist(), [[101, 102, 103]])

    def test_tail_is_padded_with_sentinel(self):
        windows = stf.build_base_windows(
            base_output_ids=[[100, 101, 102]],
            num_output_tokens=[1],
            width=5,
            device="cpu",
        )
        self.assertEqual(windows.tolist(), [[101, 102, stf.PAD_TOKEN, -1, -1]])

    def test_last_step_keeps_one_real_token(self):
        # The final decode step still needs a valid bonus at depth 0.
        windows = stf.build_base_windows(
            base_output_ids=[[100, 101, 102]],
            num_output_tokens=[2],
            width=4,
            device="cpu",
        )
        self.assertEqual(windows.tolist(), [[102, -1, -1, -1]])

    def test_rows_are_independent_per_request(self):
        windows = stf.build_base_windows(
            base_output_ids=[[10, 11, 12], [20, 21, 22]],
            num_output_tokens=[0, 2],
            width=3,
            device="cpu",
        )
        self.assertEqual(windows.tolist(), [[10, 11, 12], [22, -1, -1]])


class TestBuildForcedTargetPredicts(CustomTestCase):
    def test_tree_nodes_are_indexed_by_depth(self):
        # A topk>1 tree: root, two depth-1 siblings, two depth-2, one depth-3.
        # Only two base tokens remain past the root, so the deepest node lands
        # on the sentinel and can never be accepted.
        positions = torch.tensor([20, 21, 21, 22, 22, 23], dtype=torch.int64)
        windows = stf.build_base_windows(
            base_output_ids=[[100, 101, 102, 103]],
            num_output_tokens=[1],
            width=6,
            device="cpu",
        )
        forced = stf.build_forced_target_predicts_tree(
            windows=windows, positions=positions, bs=1, draft_token_num=6
        )
        self.assertEqual(forced.tolist(), [[101, 102, 102, 103, 103, -1]])

    def test_tree_depth_is_relative_to_each_root(self):
        # Two requests at different absolute positions must both start at depth 0.
        positions = torch.tensor([7, 8, 8, 90, 91, 92], dtype=torch.int64)
        windows = stf.build_base_windows(
            base_output_ids=[[10, 11, 12], [20, 21, 22]],
            num_output_tokens=[0, 0],
            width=3,
            device="cpu",
        )
        forced = stf.build_forced_target_predicts_tree(
            windows=windows, positions=positions, bs=2, draft_token_num=3
        )
        self.assertEqual(forced.tolist(), [[10, 11, 11], [20, 21, 22]])

    def test_chain_is_the_raw_window(self):
        forced = stf.build_forced_target_predicts_chain(
            base_output_ids=[[100, 101, 102, 103]],
            num_output_tokens=[2],
            block_size=4,
            device="cpu",
        )
        self.assertEqual(forced.tolist(), [[102, 103, -1, -1]])


class TestForceRejectAllDrafts(CustomTestCase):
    def test_tree_commits_only_the_base_token(self):
        predicts = torch.tensor([5, -1, 7, -1, 9, -1, 11, -1], dtype=torch.int32)
        accept_indices = torch.tensor(
            [[0, 1, 2, -1], [4, 5, -1, -1]], dtype=torch.int32
        )
        num_correct_drafts = torch.tensor([2, 1], dtype=torch.int32)
        target_predicts = torch.tensor([[101, 102, -1, -1], [201, -1, -1, -1]])

        stf.force_reject_all_drafts_tree(
            predicts=predicts,
            accept_indices=accept_indices,
            num_correct_drafts=num_correct_drafts,
            target_predicts=target_predicts,
        )

        self.assertEqual(predicts[0].item(), 101)
        self.assertEqual(predicts[4].item(), 201)
        self.assertEqual(num_correct_drafts.tolist(), [0, 0])
        self.assertEqual(accept_indices.tolist(), [[0, -1, -1, -1], [4, -1, -1, -1]])

    def test_tree_scrubs_the_sentinel_out_of_predicts(self):
        # A negative id would index the draft embedding table out of bounds once
        # the whole predicts buffer becomes batch.input_ids for the draft extend.
        predicts = torch.tensor([5, -1, -1, -1], dtype=torch.int32)
        accept_indices = torch.tensor([[0, -1, -1, -1]], dtype=torch.int32)
        num_correct_drafts = torch.tensor([0], dtype=torch.int32)
        target_predicts = torch.tensor([[77, -1, -1, -1]])

        stf.force_reject_all_drafts_tree(
            predicts=predicts,
            accept_indices=accept_indices,
            num_correct_drafts=num_correct_drafts,
            target_predicts=target_predicts,
        )

        self.assertGreaterEqual(int(predicts.min()), 0)
        self.assertEqual(predicts[0].item(), 77)

    def test_chain_commits_only_the_base_token(self):
        num_correct_drafts = torch.tensor([2], dtype=torch.int32)
        commit_lens = torch.tensor([3], dtype=torch.int32)
        bonus_tokens = torch.tensor([999], dtype=torch.int64)
        out_tokens = torch.tensor([[11, 12, 13, 0]], dtype=torch.int64)
        target_predicts = torch.tensor([[101, 102, 103, -1]])

        stf.force_reject_all_drafts_chain(
            num_correct_drafts=num_correct_drafts,
            commit_lens=commit_lens,
            bonus_tokens=bonus_tokens,
            out_tokens=out_tokens,
            target_predicts=target_predicts,
        )

        self.assertEqual(num_correct_drafts.tolist(), [0])
        self.assertEqual(commit_lens.tolist(), [1])
        self.assertEqual(bonus_tokens.tolist(), [101])
        self.assertEqual(out_tokens[:, 0].tolist(), [101])


class TestReadBaseOutputIds(CustomTestCase):
    def test_disabled_returns_none(self):
        reqs = [_make_req({stf.SPEC_TEACHER_FORCING_IDS_KEY: [1, 2, 3]})]
        with mock.patch.object(stf, "SPEC_TEACHER_FORCING_ENABLED", False):
            self.assertIsNone(stf.read_base_output_ids(reqs))

    def test_request_without_ids_falls_back_to_normal_path(self):
        with mock.patch.object(stf, "SPEC_TEACHER_FORCING_ENABLED", True):
            self.assertIsNone(stf.read_base_output_ids([_make_req(None)]))
            self.assertIsNone(stf.read_base_output_ids([_make_req({"other": 1})]))

    def test_partially_annotated_batch_falls_back_to_normal_path(self):
        reqs = [
            _make_req({stf.SPEC_TEACHER_FORCING_IDS_KEY: [1, 2]}),
            _make_req({}),
        ]
        with mock.patch.object(stf, "SPEC_TEACHER_FORCING_ENABLED", True):
            self.assertIsNone(stf.read_base_output_ids(reqs))

    def test_fully_annotated_batch_is_read(self):
        reqs = [
            _make_req({stf.SPEC_TEACHER_FORCING_IDS_KEY: [1, 2]}),
            _make_req({stf.SPEC_TEACHER_FORCING_IDS_KEY: [3, 4]}),
        ]
        with mock.patch.object(stf, "SPEC_TEACHER_FORCING_ENABLED", True):
            self.assertEqual(stf.read_base_output_ids(reqs), [[1, 2], [3, 4]])


class TestValidateAndClampSamplingParams(CustomTestCase):
    def _params(self, **overrides):
        params = {
            "temperature": 0,
            "max_new_tokens": 1000,
            "custom_params": {stf.SPEC_TEACHER_FORCING_IDS_KEY: [10, 11, 12, 13, 14]},
        }
        params.update(overrides)
        return params

    def test_disabled_is_a_noop(self):
        params = self._params(temperature=0.7)
        with mock.patch.object(stf, "SPEC_TEACHER_FORCING_ENABLED", False):
            stf.validate_and_clamp_sampling_params(params)
        self.assertEqual(params["max_new_tokens"], 1000)

    def test_request_without_ids_is_a_noop(self):
        params = self._params(temperature=0.7, custom_params=None)
        with mock.patch.object(stf, "SPEC_TEACHER_FORCING_ENABLED", True):
            stf.validate_and_clamp_sampling_params(params)
        self.assertEqual(params["max_new_tokens"], 1000)

    def test_max_new_tokens_is_capped_to_the_base_length(self):
        params = self._params()
        with mock.patch.object(stf, "SPEC_TEACHER_FORCING_ENABLED", True):
            stf.validate_and_clamp_sampling_params(params)
        self.assertEqual(params["max_new_tokens"], 5)

    def test_shorter_max_new_tokens_is_kept(self):
        params = self._params(max_new_tokens=3)
        with mock.patch.object(stf, "SPEC_TEACHER_FORCING_ENABLED", True):
            stf.validate_and_clamp_sampling_params(params)
        self.assertEqual(params["max_new_tokens"], 3)

    def test_missing_max_new_tokens_is_filled_in(self):
        params = self._params(max_new_tokens=None)
        with mock.patch.object(stf, "SPEC_TEACHER_FORCING_ENABLED", True):
            stf.validate_and_clamp_sampling_params(params)
        self.assertEqual(params["max_new_tokens"], 5)

    def test_non_greedy_is_rejected(self):
        # Read from the raw dict: SamplingParams.__post_init__ rewrites
        # temperature=0 into temperature=1.0 + top_k=1.
        params = self._params(temperature=0.7)
        with mock.patch.object(stf, "SPEC_TEACHER_FORCING_ENABLED", True):
            with self.assertRaisesRegex(ValueError, "temperature=0"):
                stf.validate_and_clamp_sampling_params(params)

    def test_malformed_ids_are_rejected(self):
        with mock.patch.object(stf, "SPEC_TEACHER_FORCING_ENABLED", True):
            for bad in ([], "1,2,3", [1, "two"]):
                params = self._params(
                    custom_params={stf.SPEC_TEACHER_FORCING_IDS_KEY: bad}
                )
                with self.assertRaises(ValueError):
                    stf.validate_and_clamp_sampling_params(params)


class TestValidateServerArgs(CustomTestCase):
    def _server_args(self, **overrides):
        args = dict(
            speculative_algorithm="EAGLE3",
            max_running_requests=1,
            chunked_prefill_size=-1,
            disable_radix_cache=True,
            disable_overlap_schedule=True,
        )
        args.update(overrides)
        return SimpleNamespace(**args)

    def test_disabled_accepts_anything(self):
        with mock.patch.object(stf, "SPEC_TEACHER_FORCING_ENABLED", False):
            stf.validate_server_args(self._server_args(max_running_requests=32))

    def test_conforming_config_is_accepted(self):
        with mock.patch.object(stf, "SPEC_TEACHER_FORCING_ENABLED", True):
            for algorithm in stf.SUPPORTED_ALGORITHMS:
                stf.validate_server_args(
                    self._server_args(speculative_algorithm=algorithm)
                )

    def test_unsupported_algorithm_is_rejected(self):
        with mock.patch.object(stf, "SPEC_TEACHER_FORCING_ENABLED", True):
            for algorithm in (None, "EAGLE", "STANDALONE"):
                with self.assertRaisesRegex(ValueError, "speculative-algorithm"):
                    stf.validate_server_args(
                        self._server_args(speculative_algorithm=algorithm)
                    )

    def test_nonconforming_scheduler_flags_are_rejected(self):
        offenders = [
            dict(max_running_requests=8),
            dict(chunked_prefill_size=8192),
            dict(disable_radix_cache=False),
            dict(disable_overlap_schedule=False),
        ]
        with mock.patch.object(stf, "SPEC_TEACHER_FORCING_ENABLED", True):
            for overrides in offenders:
                with self.assertRaises(ValueError):
                    stf.validate_server_args(self._server_args(**overrides))

    def test_simulate_acc_len_is_mutually_exclusive(self):
        with mock.patch.object(stf, "SPEC_TEACHER_FORCING_ENABLED", True):
            with envs.SGLANG_SIMULATE_ACC_LEN.override(3.0):
                with self.assertRaisesRegex(ValueError, "SGLANG_SIMULATE_ACC_LEN"):
                    stf.validate_server_args(self._server_args())


class TestCustomizedInfoStaging(CustomTestCase):
    def test_prefill_reserves_a_slot_and_overrides_the_token(self):
        # The result processor appends one element per request for the prefill
        # step too, so it must be filled or the whole list shifts by one.
        reqs = [_make_req({stf.SPEC_TEACHER_FORCING_IDS_KEY: [42, 43, 44]})]
        next_token_ids = torch.tensor([7], dtype=torch.int64)
        logits_output = SimpleNamespace(customized_info=None)

        with mock.patch.object(stf, "SPEC_TEACHER_FORCING_ENABLED", True):
            stf.apply_prefill_teacher_forcing(
                reqs=reqs,
                next_token_ids=next_token_ids,
                logits_output=logits_output,
            )

        self.assertEqual(next_token_ids.tolist(), [42])
        self.assertEqual(
            logits_output.customized_info,
            {stf.SPEC_TEACHER_FORCING_ACCEPT_LENGTH_KEY: [None]},
        )

    def test_prefill_is_a_noop_without_ids(self):
        reqs = [_make_req(None)]
        next_token_ids = torch.tensor([7], dtype=torch.int64)
        logits_output = SimpleNamespace(customized_info=None)

        with mock.patch.object(stf, "SPEC_TEACHER_FORCING_ENABLED", True):
            stf.apply_prefill_teacher_forcing(
                reqs=reqs,
                next_token_ids=next_token_ids,
                logits_output=logits_output,
            )

        self.assertEqual(next_token_ids.tolist(), [7])
        self.assertIsNone(logits_output.customized_info)

    def test_verify_step_stages_one_value_per_request(self):
        logits_output = SimpleNamespace(customized_info={"other": [1]})
        stf.record_accept_lengths(logits_output=logits_output, num_accept_tokens=[3, 1])
        self.assertEqual(
            logits_output.customized_info,
            {"other": [1], stf.SPEC_TEACHER_FORCING_ACCEPT_LENGTH_KEY: [3, 1]},
        )


if __name__ == "__main__":
    unittest.main()

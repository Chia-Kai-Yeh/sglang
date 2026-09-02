# Speculative Teacher Forcing

`SGLANG_SPEC_TEACHER_FORCING` turns speculative decoding into a *measurement*
mode: every verify step is scored against the output trajectory of a prior
**base** (non-speculative) run, then **all drafts are rejected** so the step
commits exactly one base token.

Because each step advances by exactly one token, every output position is
measured under the same ground-truth prefix — so EAGLE3 / DFLASH / NGRAM can be
compared on an identical trajectory instead of on the trajectory each one
happens to produce.

Implementation: `python/sglang/srt/speculative/spec_teacher_forcing.py`.

## Requirements

The server refuses to start unless all of these hold when the env var is on:

| Requirement | Why |
| --- | --- |
| `--speculative-algorithm` in `EAGLE3`, `DFLASH`, `NGRAM` | only these workers carry the prefill-token override |
| `--max-running-requests 1` | one trajectory at a time |
| `--chunked-prefill-size -1` | prefill must land in one pass |
| `--disable-radix-cache` | no prefix reuse across runs |
| `--disable-overlap-schedule` | forcing runs inline with verify |
| `SGLANG_SIMULATE_ACC_LEN` unset / `-1` | both override acceptance |

Per request, `sampling_params` must have `temperature: 0` and carry the base
ids in `custom_params["spec_teacher_forcing_ids"]`. `max_new_tokens` is capped
to `len(spec_teacher_forcing_ids)` automatically. Requests **without** that key
run the normal speculative path untouched.

## Usage

### 1. Record the base trajectory

Launch the target model with no speculative algorithm and no env var, then send
a greedy request and keep `output_ids`:

```bash
sglang serve --model-path /workspace/models/Qwen/Qwen3-4B-FP8 --max-running-requests 1
```

```python
base = requests.post(
    "http://localhost:30000/generate",
    json={"text": text, "sampling_params": {"temperature": 0, "max_new_tokens": 512}},
).json()
base_ids = base["output_ids"]
```

### 2. Replay it under a speculative algorithm

```bash
SGLANG_SPEC_TEACHER_FORCING=1 sglang serve \
    --model-path /workspace/models/Qwen/Qwen3-4B-FP8 \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path /workspace/models/AngelSlim/Qwen3-4B_eagle3 \
    --speculative-num-steps 4 \
    --speculative-eagle-topk 4 \
    --speculative-num-draft-tokens 16 \
    --max-running-requests 1 \
    --chunked-prefill-size -1 \
    --disable-radix-cache \
    --disable-overlap-schedule
```

```python
out = requests.post(
    "http://localhost:30000/generate",
    json={
        "text": text,
        "sampling_params": {
            "temperature": 0,
            "custom_params": {"spec_teacher_forcing_ids": base_ids},
        },
    },
).json()
```

### 3. Read the result

`meta_info["spec_teacher_forcing_accept_length"]` is a list aligned with
`output_ids`: one entry per output token, `None` for the prefill token, and for
each verify step the accept length that *would* have been reached (bonus token
included, so the minimum is `1`).

```python
lengths = out["meta_info"]["spec_teacher_forcing_accept_length"]
steps = [x for x in lengths if x is not None]
print(sum(steps) / len(steps))
```

Under `NGRAM` a second list comes back aligned the same way:
`meta_info["spec_teacher_forcing_ngram_match_depth"]` — for each verify step,
the deepest suffix of the current context that is present in the n-gram trie
**and has children**, i.e. the anchor that actually seeded the draft tree. `0`
means nothing usable matched. It is the explanatory variable for the accept
length above: a shallow match has little context to speculate from.

```python
depths = out["meta_info"]["spec_teacher_forcing_ngram_match_depth"]
for step, (length, depth) in enumerate(zip(lengths, depths)):
    if length is not None:
        print(step, length, depth)
```

- The ceiling is `--speculative-ngram-max-trie-depth` **minus 1** (default
  `17`): `Trie::insert()` only walks `max_trie_depth` tokens, so a node at that
  depth can never have children and can never be the reported anchor.
- External SAM corpora (`--speculative-ngram-external-corpus-path`) are **not**
  counted; the number is the online trie's depth only.
- `EAGLE3` / `DFLASH` do not emit this key at all.

Repeat step 2 for DFLASH / NGRAM with the **same** `base_ids` to compare.

## Caveats

- The generated text is the base text by construction — only the accept lengths
  are new information.
- Throughput numbers are meaningless here: each step commits one token.
- Near the end of the base sequence the comparison window runs out of real
  tokens and is padded with a sentinel that never matches, so the last
  `--speculative-num-draft-tokens` steps are biased low. Drop that tail before
  averaging if it matters.

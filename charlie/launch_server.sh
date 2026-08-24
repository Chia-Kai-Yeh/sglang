#!/bin/bash
set -euo pipefail

COMMON=(
    --model-path /workspace/models/Qwen/Qwen3-4B-FP8
    --mem-fraction-static 0.9
    --max-running-requests 1
    --cuda-graph-max-bs-decode 1
)

case ${1:-base} in
    base)
        sglang serve "${COMMON[@]}"
        ;;
    eagle3)
        sglang serve "${COMMON[@]}" \
            --speculative-algorithm EAGLE3 \
            --speculative-draft-model-path /workspace/models/AngelSlim/Qwen3-4B_eagle3 \
            --speculative-num-steps 4 \
            --speculative-eagle-topk 4 \
            --speculative-num-draft-tokens 16
        ;;
    dflash)
        sglang serve "${COMMON[@]}" \
            --speculative-algorithm DFLASH \
            --speculative-draft-model-path /workspace/models/z-lab/Qwen3-4B-DFlash-b16 \
            --speculative-num-draft-tokens 16
        ;;
    ngram)
        sglang serve "${COMMON[@]}" \
            --speculative-algorithm NGRAM \
            --speculative-num-draft-tokens 16
        ;;
    *)
        exit 1
        ;;
esac

#!/bin/bash
set -euo pipefail

COMMON=(
    --model Qwen/Qwen3-4B-FP8
    --mem-fraction-static 0.85
    --max-running-requests 1
    --cuda-graph-max-bs-decode 1
    --disable-prefill-cuda-graph
)

case "${1:-base}" in
    base)
        sglang serve "${COMMON[@]}"
        ;;
    eagle3)
        sglang serve "${COMMON[@]}" \
            --speculative-algorithm EAGLE3 \
            --speculative-draft-model AngelSlim/Qwen3-4B_eagle3 \
            --speculative-num-steps 5 \
            --speculative-eagle-topk 4 \
            --speculative-num-draft-tokens 16 \
            --speculative-draft-model-quantization fp8
        ;;
    dflash)
        sglang serve "${COMMON[@]}" \
            --speculative-algorithm DFLASH \
            --speculative-draft-model z-lab/Qwen3-4B-DFlash-b16 \
            --speculative-num-draft-tokens 16 \
            --speculative-draft-model-quantization fp8
        ;;
    domino)
        sglang serve "${COMMON[@]}" \
            --speculative-algorithm DFLASH \
            --speculative-draft-model Huang2020/Qwen3-4B-Domino-b16 \
            --speculative-num-draft-tokens 16 \
            --speculative-draft-model-quantization fp8
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

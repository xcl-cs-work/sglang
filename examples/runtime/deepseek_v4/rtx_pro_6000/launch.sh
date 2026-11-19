#!/usr/bin/env bash
# Single-node, text-only DeepSeek-V4.1-Flash-1017 on eight 96 GB SM120 GPUs.
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to the local 1017 checkpoint directory}"
source "$(dirname "$0")/env.sh"

command=(python "$(dirname "$0")/serve.py")
if [[ -n "${PREFERRED_NUMA_NODE:-}" ]]; then
    export SGLANG_AUTO_NUMA_BIND=0
    command=(numactl "--preferred=$PREFERRED_NUMA_NODE" "${command[@]}")
fi
exec "${command[@]}" \
    --model-path "$MODEL_PATH" --served-model-name deepseek-v4.1-flash-1017 \
    --trust-remote-code --language-model-only --load-format safetensors \
    --model-loader-extra-config '{"enable_multithread_load":false}' \
    --tp-size 8 --ep-size 8 --moe-runner-backend marlin --disable-shared-experts-fusion \
    --fp8-gemm-backend triton --sampling-backend pytorch --dsv4-attn-backend flashmla \
    --disable-custom-all-reduce --enforce-disable-flashinfer-allreduce-fusion \
    --context-length 32768 --max-total-tokens 65536 --max-running-requests 8 \
    --chunked-prefill-size 1024 --mem-fraction-static 0.85 \
    --cuda-graph-backend-prefill disabled --cuda-graph-backend-decode disabled \
    --reasoning-parser auto --tool-call-parser auto --watchdog-timeout 1200 \
    --host "${HOST:-127.0.0.1}" --port "${PORT:-31041}" "$@"

#!/usr/bin/env bash
# Single-node, text-only DeepSeek-V4.1-Flash-1017 on four or eight 96 GB SM120 GPUs.
set -euo pipefail

: "${MODEL_PATH:?Set MODEL_PATH to the local 1017 checkpoint directory}"
source "$(dirname "$0")/env.sh"

gpu_count=${GPU_COUNT:-8}
case "$gpu_count" in
    4)
        profile=(--max-total-tokens 131072 --max-running-requests 128 --mem-fraction-static 0.90
                 --cuda-graph-max-bs-decode 128 --cuda-graph-bs-decode 1 2 4 8 16 32 64 128)
        # Prefer the memory socket near each GPU for its pinned host tables.
        export SGLANG_AUTO_NUMA_BIND=1 SGLANG_NUMA_BIND_V2=0
        # The tested 0.5.23 wheel predates --engram-config.
        has_engram_config=$(python -c 'from importlib.util import find_spec; print(find_spec("sglang.srt.arg_groups.engram_config") is not None)')
        if [[ "$has_engram_config" == True ]]; then
            profile+=(--engram-config '{"placement":"cpu"}')
        else
            export SGLANG_ENABLE_DSV41_ENGRAM_HOST_TABLE=1
            export SGLANG_DSV41_ENGRAM_HOST_TABLE_LAYOUT=per_rank
            export SGLANG_OPT_DSV41_ENGRAM_PEER_SHARDS=0
        fi
        ;;
    8)
        profile=(--max-total-tokens 65536 --max-running-requests 8 --mem-fraction-static 0.85
                 --cuda-graph-backend-decode disabled)
        ;;
    *)
        echo "GPU_COUNT must be 4 or 8" >&2
        exit 2
        ;;
esac

command=(python "$(dirname "$0")/serve.py")
if [[ -n "${PREFERRED_NUMA_NODE:-}" ]]; then
    export SGLANG_AUTO_NUMA_BIND=0
    command=(numactl "--preferred=$PREFERRED_NUMA_NODE" "${command[@]}")
fi
exec "${command[@]}" \
    --model-path "$MODEL_PATH" --served-model-name deepseek-v4.1-flash-1017 \
    --trust-remote-code --language-model-only --load-format safetensors \
    --model-loader-extra-config '{"enable_multithread_load":false}' \
    --tp-size "$gpu_count" --ep-size "$gpu_count" --moe-runner-backend marlin --disable-shared-experts-fusion \
    --fp8-gemm-backend triton --sampling-backend pytorch --dsv4-attn-backend flashmla \
    --disable-custom-all-reduce --enforce-disable-flashinfer-allreduce-fusion \
    --context-length 32768 --chunked-prefill-size 1024 \
    --cuda-graph-backend-prefill disabled "${profile[@]}" \
    --reasoning-parser auto --tool-call-parser auto --watchdog-timeout 1200 \
    --host "${HOST:-127.0.0.1}" --port "${PORT:-31041}" "$@"

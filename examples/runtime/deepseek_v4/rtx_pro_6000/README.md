# DeepSeek-V4.1-Flash-1017 on eight RTX PRO 6000 GPUs

This example serves the trained, text-only model on a single PCIe node with eight
96 GB Blackwell Server Edition GPUs (SM120). It uses TP8/EP8, native FP4 experts
with Marlin, Triton attention, PyTorch sampling, and GPU-resident Engram tables.
The server accepts 32,768 tokens per request, with up to eight concurrent requests
and a 65,536-token cap on the shared KV cache.

The verified deployment used the official `deepseek-ai/DeepSeek-V4.1-Flash-1017`
checkpoint at revision `08c8657f02291e11b6f6b63e0e16014f8e748518`, driver
590.48.01, PyTorch 2.14.1+cu130 and SGLang 0.5.23.post99+miles.g18a44326f6.
That wheel needs the SM120 indexer fallback in [serve.py](serve.py). The wrapper
defers to SGLang versions that already distinguish the SM100 sparse indexer.

## Launch

Run these commands in this example directory on the allocated GPU node, with
the CUDA-enabled SGLang Python environment active and its CUDA toolchain on
`PATH`. Set `MODEL_PATH` to the directory containing the checkpoint index,
safetensors, config, and tokenizer.

```bash
export MODEL_PATH=/path/to/DeepSeek-V4.1-Flash-1017
export LOCAL_CACHE_DIR="$(mktemp -d /tmp/deepseek-1017-XXXXXXXX)"
source ./env.sh
python precompile.py
./launch.sh
```

Reuse the same private `LOCAL_CACHE_DIR` for subsequent launches. Precompilation
builds both Marlin expert-parallel specializations before the first request.
The default endpoint is `http://127.0.0.1:31041/v1`. Set `HOST` and `PORT`
before launch to serve on a different interface or port.

If worker startup exhausts RAM on one NUMA node, inspect
`numactl --hardware` and set `PREFERRED_NUMA_NODE` to a node with available
memory. The tested machine needed `PREFERRED_NUMA_NODE=1`.

## Optional node-local checkpoint staging

Use a normal local checkpoint directory if the disk has room. When the 475 GiB
checkpoint exceeds local disk capacity, [stage_model.py](stage_model.py) can
hold each shard in a sealed Linux memory file. The files appear under a private
scratch directory through `/proc/<pid>/fd` links. The staging process and server
must run as the same user on the same node with access to those links.
Budget host RAM for the staged checkpoint plus the runtime inside the job's
memory limit. The default host headroom check is 160 GiB.

From the same shell, after setting `MODEL_PATH` and `LOCAL_CACHE_DIR` above:

```bash
python stage_model.py "$MODEL_PATH" "$LOCAL_CACHE_DIR/model" &
stage_pid=$!
until test -f "$LOCAL_CACHE_DIR/model/STAGED"; do
    kill -0 "$stage_pid" || { wait "$stage_pid"; exit 1; }
    sleep 5
done
export MODEL_PATH="$LOCAL_CACHE_DIR/model"
./launch.sh
```

Keep the staging process alive across server restarts. When finished, stop the
server first, then run `kill "$stage_pid"` to release the checkpoint RAM.
Staging took about 72 minutes over the tested node's 1 Gbit/s link. Once staged,
each rank loaded its share of the real weights in about 25 seconds.

## Verify sampling and measure concurrency

Run the client on a host that can reach the server. The default check sends
three turns of a fictional support handoff, including a correction to the
affected region. It sends the actual assistant replies in the next request and
checks the final answer against the updated incident details.

```bash
python check_server.py --base-url http://127.0.0.1:31041/v1 --reasoning-effort low
python check_server.py --base-url http://127.0.0.1:31041/v1 --benchmark --repetitions 2
```

The optional benchmark streams up to 128 output tokens per request, at
concurrency 1, 2, 4 and 8. It divides the total completion tokens reported by
the API by the wall time from the first request starting to the last stream
finishing. The tested server delivered about 12, 24, 48 and 93 aggregate
output tokens/s respectively on short prompts, with reasoning disabled.

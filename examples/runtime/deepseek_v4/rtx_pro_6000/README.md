# DeepSeek-V4.1-Flash-1017 on four or eight RTX PRO 6000 GPUs

This example serves the trained, text-only model on a single PCIe node with
96 GB Blackwell Server Edition GPUs (SM120). Both profiles keep routed experts
in native FP4, use Marlin, Triton attention and PyTorch sampling, and support
32,768 tokens per request.

| GPUs | Tensor / expert parallelism | Engram placement | Total KV token cap | Max active requests | Decode graphs |
| --- | --- | --- | ---: | ---: | --- |
| 8 (default) | TP8 / EP8 | GPU | 65,536 | 8 | Disabled |
| 4 | TP4 / EP4 | Pinned host RAM, sharded per rank | 131,072 | 128 | Batch sizes 1, 2, 4, 8, 16, 32, 64, 128 |

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

To use the four-GPU profile, run `GPU_COUNT=4 ./launch.sh` in place of
`./launch.sh`. Budget about 189 GiB of host RAM for Engram (47.2 GiB per
worker), in addition to any checkpoint staging and loader memory. The tested
four-GPU job reserved 128 CPUs and about 756 GiB of RAM. The launcher asks
each worker to prefer the NUMA node local to its GPU.

Reuse the same private `LOCAL_CACHE_DIR` for subsequent launches. Precompilation
builds both Marlin expert-parallel specializations before the first request.
The default endpoint is `http://127.0.0.1:31041/v1`. Set `HOST` and `PORT`
before launch to serve on a different interface or port.

If worker startup exhausts RAM on one NUMA node, inspect
`numactl --hardware` and set `PREFERRED_NUMA_NODE` to a node with available
memory. The tested eight-GPU node used `PREFERRED_NUMA_NODE=1`. Leave the variable
unset for the four-GPU profile to use each worker’s GPU-local preference.

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

On the four-GPU node the two scratch filesystems could hold only part of the
checkpoint. The stager can copy whole shards there and keep the remainder in
RAM. For example, on the allocated node:

```bash
python stage_model.py "$MODEL_PATH" "$LOCAL_CACHE_DIR/model" \
    --disk-dir /var/tmp 50 --disk-dir /tmp 12 --headroom-gib 240
```

This is a foreground staging command. Use a second shell on the same node to
launch once `STAGED` exists, or use the background pattern above. Choose disk
budgets that leave room for the runtime and JIT caches. The measured split was
58.9 GiB on disk and 416.3 GiB in RAM. With the checkpoint staged, loading the
four GPU workers and their host tables took 16 min 47 s. After KV allocation
and graph capture there were 13.1 GiB free per GPU.

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
finishing. The eight-GPU profile delivered about 12, 24, 48 and 93 aggregate
output tokens/s respectively on short prompts, with reasoning disabled.

For the four-GPU profile, run the benchmark with
`--concurrency 1 8 32 64 128 --repetitions 2`. The measured aggregate rates
were **23, 206, 407, 492 and 1,068 output tokens/s**, respectively, with short
prompts and 128 output tokens per request. At 128 concurrent requests median
time to first token was 2.09 s. The two profiles use different decode-graph
and batching settings; these results are measurements of each configuration.

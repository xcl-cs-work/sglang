"""Use the portable V4.1 indexer on SM120 with the older serving wheel."""

import runpy

import torch
from sglang.srt.layers.attention.dsv4 import v41_indexer

# The tested 0.5.23.post99 wheel selected SM100-only sparse DeepGEMM on SM120.
# Current SGLang has a separate is_sm100 gate for that path.
if not hasattr(v41_indexer, "is_sm100"):
    original_gate = v41_indexer.is_sm100_or_newer

    def supports_indexer():
        return original_gate() and torch.cuda.get_device_capability()[0] != 12

    v41_indexer.is_sm100_or_newer = supports_indexer

if __name__ == "__main__":
    runpy.run_module("sglang.launch_server", run_name="__main__")

"""Build the two BF16 Marlin EP specializations used by the server."""

import torch
from sglang.kernels.ops.moe.moe_wna16_marlin import _jit_moe_wna16_marlin_module

if __name__ == "__main__":
    for zero_inactive in (False, True):
        _jit_moe_wna16_marlin_module(
            torch.bfloat16, True, False, False, zero_inactive, False
        )
    values = torch.tensor([2.0, -1.0], device="cuda")
    torch.testing.assert_close(torch.exp2(values), values.new_tensor([4.0, 0.5]))

"""LoRA for Inkling 2 (prototype): the stacked-source qkvr wrapper and the routed-expert side path.

Expert adapters are shared-outer (one gate/up lora_A and one down lora_B for every expert), so the side path is:
  S = x @ A_gu^T                            dense, once per token, [T, 2R] (gate | up halves, max-rank spaced)
  dgu[row] = B_gu[e] @ S[token]             per routed row, added to the gate/up accumulator before the SwiGLU
  u[row] = A_dn[e] @ h[row]                 per routed row; h is the BF16 activated hidden (gamma applied)
  out += (sum_k u[token, k]) @ B_dn^T       dense, once per token
Pool buffers: gate_up A [slots, 1, 2R, D], gate_up B [slots, E, 2F, R] (rows interleaved like the served w13 at
load), down A [slots, E, R, F], down B [slots, 1, D, R]; MoE B factors are pre-scaled by the pool.
The routed-row gathers make this a correctness prototype, not a fast path.
"""

from __future__ import annotations

import torch

from sglang.srt.lora.layers import BaseLayerWithLoRA, MergedColumnParallelLinearWithLoRA
from sglang.srt.lora.utils import capturing_lora_graph

# routed rows per gather chunk: bounds the [rows, 2F, R] / [rows, R, F] factor gathers
_ROW_CHUNK = 2048


class Inkling2QKVRLinearWithLoRA(MergedColumnParallelLinearWithLoRA):
    """qkvr runs once on the stacked [n T, H] Q/K/V sources; the LoRA batch info covers T rows, so the delta is
    applied per T-row chunk into the matching rows of the base output."""

    is_inkling2_qkvr_lora = True

    def forward(self, input_: torch.Tensor, lora_num_sources: int = 1):
        if lora_num_sources == 1 or not self.lora_active:
            return super().forward(input_)
        base = self.base_layer.quant_method.apply(
            self.base_layer, input_, self.base_layer.bias
        )
        tokens = input_.shape[0] // lora_num_sources
        for c in range(lora_num_sources):
            rows = slice(c * tokens, (c + 1) * tokens)
            self.apply_lora(base[rows], input_[rows])
        return base, None


def _token_slots(batch_info, tokens: int, device) -> torch.Tensor:
    """Adapter slot of each token in the forward's own order (-1: none), from the dense LoRA batch info,
    without a host sync (CUDA-graph safe)."""
    segments = batch_info.seg_indptr[1 : batch_info.num_segments + 1]
    rows = torch.arange(tokens, device=device, dtype=segments.dtype)
    segment = torch.searchsorted(segments, rows, right=True).clamp_(
        max=batch_info.num_segments - 1
    )
    slots = batch_info.weight_indices[segment].long()
    if batch_info.permutation is not None:
        slots = torch.empty_like(slots).scatter_(
            0, batch_info.permutation[:tokens].long(), slots
        )
    ranks = batch_info.lora_ranks[slots.clamp(min=0)]
    return torch.where((slots >= 0) & (ranks > 0), slots, -1)


class Inkling2ExpertsLoRA:
    """Adapter state of one Inkling2Experts; the expert forward paths call begin / gate_up / down."""

    def __init__(self, experts, lora_backend):
        if experts.tp_size != 1:
            raise NotImplementedError(
                "Inkling 2 expert LoRA prototype: expert TP1 only (EP is fine)"
            )
        self.experts = experts
        self.num_local = experts.w13_weight.shape[0]
        self.start = experts.ep_rank * self.num_local
        self.offset = 0
        self.backend = lora_backend
        self.buffers = None

    def set_buffers(self, gate_up_a, gate_up_b, down_a, down_b):
        if gate_up_a.shape[1] != 1 or down_b.shape[1] != 1:
            raise ValueError(
                "Inkling 2 expert LoRA needs shared-outer adapters (--experts-shared-outer-loras)"
            )
        self.buffers = (gate_up_a[:, 0], gate_up_b, down_a, down_b[:, 0])
        # the pool keys per-expert factors by local id under EP when it localizes, else by global id
        self.offset = self.start if gate_up_b.shape[1] == self.num_local else 0

    def active(self) -> bool:
        if self.buffers is None:
            return False
        info = self.backend.get_batch_info()
        if info is None:
            return False
        return bool(getattr(info, "has_active_lora", True)) or capturing_lora_graph()

    def begin(self, x: torch.Tensor, ids: torch.Tensor):
        """Per-forward context, or None when no adapter runs."""
        if not self.active():
            return None
        return _ExpertLoRAForward(self, x, ids)


class _ExpertLoRAForward:
    def __init__(self, lora: Inkling2ExpertsLoRA, x, ids):
        a_gu, self.b_gu, self.a_dn, b_dn = lora.buffers
        self.b_dn = b_dn
        self.tokens, self.top_k = ids.shape
        self.ids = ids.flatten().long()
        slots = _token_slots(lora.backend.get_batch_info(), self.tokens, x.device)
        self.valid = slots >= 0
        self.slot = slots.clamp(min=0)
        # S = x @ A_gu[slot]^T, once per token, for every slot (max_loras_per_batch is small)
        per_slot = torch.einsum("td,srd->str", x, a_gu)  # [slots, T, 2R]
        self.s = per_slot[self.slot, torch.arange(self.tokens, device=x.device)].float()
        self.s.mul_(self.valid[:, None])
        self.rank = a_gu.shape[1] // 2
        self.start, self.num_local, self.offset = (
            lora.start,
            lora.num_local,
            lora.offset,
        )

    def _rows(self, row_src):
        """token, slot, factor index and this-rank mask of routed pairs row_src."""
        token = torch.div(row_src, self.top_k, rounding_mode="floor")
        ids = self.ids[row_src]
        local = (ids >= self.start) & (ids < self.start + self.num_local)
        index = (ids - self.offset).clamp(0, self.b_gu.shape[1] - 1)
        return token, self.slot[token], index, local

    def gate_up(
        self,
        gate_up: torch.Tensor,
        row_src: torch.Tensor,
        scale: torch.Tensor | None = None,
    ):
        """Add the gate/up delta to accumulator rows (row r is routed pair row_src[r] = token * top_k + k).
        scale: per-row factor from accumulator units to true units (NVFP4 raw accumulators), else None."""
        r = self.rank
        out = gate_up
        for start in range(0, row_src.numel(), _ROW_CHUNK):
            rows = slice(start, start + _ROW_CHUNK)
            token, slot, expert, local = self._rows(row_src[rows])
            b = self.b_gu[
                slot, expert
            ].float()  # [m, 2F, R], rows interleaved (gate f, up f)
            b = b.view(b.shape[0], -1, 2, r)
            s = self.s[token].view(-1, 2, r)
            delta = torch.einsum("mfcr,mcr->mfc", b, s).flatten(1)
            if scale is not None:
                delta = delta / scale[rows, None]
            # rows of other ranks' experts are left as they are (their scale may be unset)
            delta = torch.where(local[:, None], delta, 0.0)
            out[rows] = (out[rows].float() + delta).to(out.dtype)
        return out

    def down(self, hidden: torch.Tensor, row_src: torch.Tensor) -> torch.Tensor:
        """[T, D] fp32 down delta of this rank's experts from the activated rows (gamma already applied);
        under EP it is a partial sum, added before the expert combine."""
        # each routed pair owns one row: scatter by pair, then sum the top-k (deterministic, no atomics)
        u_pairs = torch.zeros(
            self.tokens * self.top_k,
            self.rank,
            device=hidden.device,
            dtype=torch.float32,
        )
        for start in range(0, row_src.numel(), _ROW_CHUNK):
            rows = slice(start, start + _ROW_CHUNK)
            token, slot, expert, local = self._rows(row_src[rows])
            a = self.a_dn[slot, expert].float()  # [m, R, F]
            u = torch.bmm(a, hidden[rows].float().unsqueeze(-1)).squeeze(-1)
            u_pairs[row_src[rows]] = torch.where(local[:, None], u, 0.0)
        u_sum = u_pairs.view(self.tokens, self.top_k, self.rank).sum(1)
        u_sum.mul_(self.valid[:, None])
        delta = torch.zeros(
            self.tokens, self.b_dn.shape[1], device=hidden.device, dtype=torch.float32
        )
        for s in range(self.b_dn.shape[0]):
            delta += (self.slot == s)[:, None] * (u_sum @ self.b_dn[s].float().T)
        return delta


def interleave_gate_up_rows(weights):
    """[E, 2F, R] gate || up rows (the adapter loader's stacking) -> the served w13 interleave (2f gate, 2f+1 up)."""
    if isinstance(weights, dict):
        return {k: interleave_gate_up_rows(v) for k, v in weights.items()}
    *lead, two_f, r = weights.shape
    return (
        weights.view(*lead, 2, two_f // 2, r)
        .transpose(-3, -2)
        .reshape(*lead, two_f, r)
        .contiguous()
    )


def is_lora_wrapped(module) -> bool:
    return isinstance(module, BaseLayerWithLoRA)

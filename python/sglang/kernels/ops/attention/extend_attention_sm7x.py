"""Extend (prefill) attention for SM7x on FP16 tensor cores.

Triton emits no tensor-core MMA below SM80, so its extend kernel runs on CUDA cores
there. This path computes the same attention as extend_attention_fwd for causal
self-attention with an optional sliding window and relative-bias score_mod
(Aux0[q_idx, head, q_pos - kv_pos] for 0 <= q_pos - kv_pos < Aux0.shape[2]):

* keys every query of a block sees beyond the bias range go through PyTorch's
  memory-efficient SDPA, which returns its log-sum-exp;
* the near band (bias, causal and window masks) is an explicit FP16 GEMM pair;
* the two partial softmaxes are merged by their log-sum-exps.

The BF16 operands are converted to FP16, which SM7x tensor cores take: Inkling's
q and k are L2-clipped and v is RMS-normalized, well inside the FP16 range.
Positions are relative to each request's kv_indices list, as in the Triton kernel.
"""

import torch

from sglang.kernels.ops.attention.score_mod import relative_bias_score_mod

BLOCK_Q = 1024


def supports(
    *,
    custom_mask,
    is_causal,
    logit_cap,
    sinks,
    xai_temperature_len,
    k_scale,
    v_scale,
    score_mod,
    q_extend,
):
    return (
        custom_mask is None
        and is_causal
        and logit_cap <= 0
        and sinks is None
        and xai_temperature_len <= 0
        and k_scale == 1.0
        and v_scale == 1.0
        and score_mod in (None, relative_bias_score_mod)
        and q_extend.dtype in (torch.bfloat16, torch.float16)
    )


def _token_rows(buffer, kv_heads):
    # [slots, kv_heads, D] or paged [pages, page_size, kv_heads, D].
    return buffer if buffer.ndim == 3 else buffer.flatten(0, 1)


def _far_attention(q, k, v, sm_scale):
    """Unmasked attention of q [KVH, G, nb, D] over k, v [F, KVH, D] (FP16):
    output [KVH, G, nb, D] FP32 and natural-log LSE [KVH, G, nb]."""
    kv_heads, group, rows, _ = q.shape
    outs, lses = [], []
    for head in range(kv_heads):
        keys = k[:, head][None, None].expand(1, group, -1, -1)
        values = v[:, head][None, None].expand(1, group, -1, -1)
        out, lse, *_ = torch.ops.aten._scaled_dot_product_efficient_attention(
            q[head][None], keys, values, None, True, scale=sm_scale
        )
        outs.append(out[0].float())
        lses.append(lse[0, :, :rows])
    return torch.stack(outs), torch.stack(lses)


def _near_attention(q, k, v, q_pos, k_pos, sm_scale, bias, bias_len, window):
    """Masked attention of q [KVH, G, nb, D] over the band k, v [nk, KVH, D] (FP16)
    with the relative bias [nb, KVH * G, bias_len]: FP32 output and LSE."""
    kv_heads, group, rows, dim = q.shape
    scores = torch.bmm(
        q.reshape(kv_heads, group * rows, dim),
        k.permute(1, 2, 0),
        out_dtype=torch.float32,
    ).view(kv_heads, group, rows, -1)
    scores *= sm_scale
    dist = q_pos[:, None] - k_pos[None, :]
    if bias is not None:
        index = dist.clamp(0, bias_len - 1)[:, None, :].expand(-1, bias.shape[1], -1)
        rel = torch.gather(bias, 2, index).float()
        rel = rel.masked_fill(~((dist >= 0) & (dist < bias_len))[:, None, :], 0.0)
        scores += rel.permute(1, 0, 2).view(kv_heads, group, rows, -1)
    visible = dist >= 0
    if window is not None:
        visible &= dist <= window
    scores.masked_fill_(~visible, float("-inf"))
    lse = torch.logsumexp(scores, dim=-1)
    probs = torch.exp(scores - lse[..., None]).to(torch.float16)
    out = torch.bmm(
        probs.view(kv_heads, group * rows, -1),
        v.permute(1, 0, 2),
        out_dtype=torch.float32,
    ).view(kv_heads, group, rows, dim)
    return out, lse


def _request_attention(q, k, v, prefix_len, rel_bias, sm_scale, window, out, lse_out):
    """One request: q [n, H, D] at positions prefix_len + i; k, v [prefix_len + n, KVH, D]."""
    rows, heads, dim = q.shape
    kv_heads = k.shape[1]
    group = heads // kv_heads
    bias_len = rel_bias.shape[2] if rel_bias is not None else 0
    q = q.to(torch.float16).view(rows, kv_heads, group, dim).permute(1, 2, 0, 3)
    k, v = k.to(torch.float16), v.to(torch.float16)
    positions = torch.arange(prefix_len + rows, device=q.device)
    for start in range(0, rows, BLOCK_Q):
        end = min(rows, start + BLOCK_Q)
        q_lo, q_hi = prefix_len + start, prefix_len + end
        # Keys every query of the block sees with no bias and no mask.
        far_end = 0 if window is not None else max(0, q_lo - max(bias_len, 1) + 1)
        near_lo = max(far_end, q_lo - window) if window is not None else far_end
        block_q = q[:, :, start:end].contiguous()
        block_out, block_lse = _near_attention(
            block_q,
            k[near_lo:q_hi],
            v[near_lo:q_hi],
            positions[q_lo:q_hi],
            positions[near_lo:q_hi],
            sm_scale,
            rel_bias[start:end] if rel_bias is not None else None,
            bias_len,
            window,
        )
        if far_end > 0:
            far_out, far_lse = _far_attention(
                block_q, k[:far_end], v[:far_end], sm_scale
            )
            lse = torch.logaddexp(block_lse, far_lse)
            block_out = (
                block_out * torch.exp(block_lse - lse)[..., None]
                + far_out * torch.exp(far_lse - lse)[..., None]
            )
            block_lse = lse
        out[start:end] = block_out.permute(2, 0, 1, 3).reshape(end - start, heads, dim)
        if lse_out is not None:
            lse_out[start:end] = block_lse.permute(2, 0, 1).reshape(end - start, heads)


def extend_attention_sm7x(
    q_extend,
    k_extend,
    v_extend,
    o_extend,
    k_buffer,
    v_buffer,
    qo_indptr,
    kv_indptr,
    kv_indices,
    sm_scale,
    sliding_window_size=-1,
    aux_tensors=None,
    lse_extend=None,
):
    """extend_attention_fwd for the inputs supports() accepts. Syncs on the
    indptrs; not CUDA-graph safe."""
    kv_heads = k_extend.shape[1]
    k_cache, v_cache = _token_rows(k_buffer, kv_heads), _token_rows(v_buffer, kv_heads)
    window = sliding_window_size if sliding_window_size > 0 else None
    rel_bias = aux_tensors[0] if aux_tensors is not None else None
    qo, kv = qo_indptr.tolist(), kv_indptr.tolist()
    for i in range(len(qo) - 1):
        qs, qe, ps, pe = qo[i], qo[i + 1], kv[i], kv[i + 1]
        if qe == qs:
            continue
        indices = kv_indices[ps:pe].long()
        k = torch.cat([k_cache[indices], k_extend[qs:qe]])
        v = torch.cat([v_cache[indices], v_extend[qs:qe]])
        _request_attention(
            q_extend[qs:qe],
            k,
            v,
            pe - ps,
            rel_bias[qs:qe] if rel_bias is not None else None,
            sm_scale,
            window,
            o_extend[qs:qe],
            lse_extend[qs:qe] if lse_extend is not None else None,
        )

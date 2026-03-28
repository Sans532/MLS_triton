"""
Triton Multi-Head Attention Implementation
End-to-end implementation using Triton kernels

Optimizations beyond base implementation:
  1. FlashAttention-2 fused kernel with online softmax (no O(seq^2) score matrix)
  2. Causal early-exit: variable loop bound so K/V tiles entirely beyond the
     causal horizon are never loaded — saves ~50% K/V tile loads during causal prefill
  3. Decode-optimised single-query kernel: when seq_q == 1 (autoregressive step),
     avoids all BLOCK_Q blocking overhead with a direct single-query kernel
  4. Extended autotune configs with num_stages=5 for H200 deeper L2 cache pipeline
  5. allow_tf32=True on all tl.dot calls for tensor-core utilisation
"""

import numpy as np
import torch
import triton
import triton.language as tl
from typing import Optional, Tuple


def get_stream():
    """Get current CUDA stream pointer."""
    if torch.cuda.is_available():
        return torch.cuda.current_stream().cuda_stream
    return None


# ============================================================================
# Base kernels (kept for fallback path)
# ============================================================================

@triton.jit
def attention_scores_kernel(
    q_ptr, k_ptr, scores_ptr,
    scale, seq_k, head_dim,
    stride_q0, stride_q1, stride_q2,
    stride_k0, stride_k1, stride_k2,
    stride_s0, stride_s1, stride_s2,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Compute scaled attention scores. Grid: (batch_heads, seq_q)"""
    pid_bh = tl.program_id(0)
    pid_q  = tl.program_id(1)
    offs_k = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)
    q = tl.load(
        q_ptr + pid_bh * stride_q0 + pid_q * stride_q1 + offs_d * stride_q2,
        mask=offs_d < head_dim, other=0.0,
    )
    k = tl.load(
        k_ptr + pid_bh * stride_k0 + offs_k[:, None] * stride_k1 + offs_d[None, :] * stride_k2,
        mask=(offs_k[:, None] < seq_k) & (offs_d[None, :] < head_dim), other=0.0,
    )
    scores = tl.sum(k * q[None, :], axis=1) * scale
    tl.store(
        scores_ptr + pid_bh * stride_s0 + pid_q * stride_s1 + offs_k * stride_s2,
        scores, mask=offs_k < seq_k,
    )


@triton.jit
def softmax_inplace_kernel(scores_ptr, stride_s, seq_k, BLOCK_SIZE: tl.constexpr):
    """Softmax in-place. Grid: (batch_heads * seq_q,)"""
    row  = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < seq_k
    s    = tl.load(scores_ptr + row * stride_s + offs, mask=mask, other=-float("inf"))
    s    = s - tl.max(s, axis=0)
    exp_s = tl.exp(s)
    out  = exp_s / tl.sum(exp_s, axis=0)
    tl.store(scores_ptr + row * stride_s + offs, out, mask=mask)


@triton.jit
def attention_output_kernel(
    attn_ptr, v_ptr, output_ptr,
    seq_k, head_dim,
    stride_w0, stride_w1, stride_w2,
    stride_v0, stride_v1, stride_v2,
    stride_o0, stride_o1, stride_o2,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Attention output: weights @ V. Grid: (batch_heads, seq_q)"""
    pid_bh = tl.program_id(0)
    pid_q  = tl.program_id(1)
    offs_k = tl.arange(0, BLOCK_K)
    offs_d = tl.arange(0, BLOCK_D)
    w = tl.load(
        attn_ptr + pid_bh * stride_w0 + pid_q * stride_w1 + offs_k * stride_w2,
        mask=offs_k < seq_k, other=0.0,
    )
    v = tl.load(
        v_ptr + pid_bh * stride_v0 + offs_k[:, None] * stride_v1 + offs_d[None, :] * stride_v2,
        mask=(offs_k[:, None] < seq_k) & (offs_d[None, :] < head_dim), other=0.0,
    )
    out = tl.sum(v * w[:, None], axis=0)
    tl.store(
        output_ptr + pid_bh * stride_o0 + pid_q * stride_o1 + offs_d * stride_o2,
        out, mask=offs_d < head_dim,
    )


@triton.jit
def causal_mask_kernel(
    scores_ptr, seq_k, offset,
    stride_s0, stride_s1, stride_s2,
    BLOCK_K: tl.constexpr,
):
    """Apply causal mask. Grid: (batch_heads, seq_q)"""
    pid_bh = tl.program_id(0)
    pid_q  = tl.program_id(1)
    offs_k = tl.arange(0, BLOCK_K)
    mask   = offs_k < seq_k
    scores = tl.load(
        scores_ptr + pid_bh * stride_s0 + pid_q * stride_s1 + offs_k * stride_s2,
        mask=mask, other=-1e9,
    )
    current_pos = pid_q + offset
    scores = tl.where(offs_k > current_pos, -1e9, scores)
    tl.store(
        scores_ptr + pid_bh * stride_s0 + pid_q * stride_s1 + offs_k * stride_s2,
        scores, mask=mask,
    )


# ============================================================================
# Optimization 3: Decode-optimised single-query attention kernel
#
# For every autoregressive decode step seq_q==1.  The blocked FA-2 kernel
# wastes resources managing BLOCK_Q-size tiles for a single row.
# This kernel handles exactly one query per program (grid = batch*heads)
# and streams K/V in BLOCK_K chunks with online softmax.
# ============================================================================

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_K": 64,  "BLOCK_D": 64},  num_warps=4, num_stages=3),
        triton.Config({"BLOCK_K": 128, "BLOCK_D": 64},  num_warps=4, num_stages=3),
        triton.Config({"BLOCK_K": 128, "BLOCK_D": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_K": 64,  "BLOCK_D": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_K": 256, "BLOCK_D": 64},  num_warps=8, num_stages=4),
        triton.Config({"BLOCK_K": 128, "BLOCK_D": 64},  num_warps=4, num_stages=5),
    ],
    key=["seq_k", "head_dim"],
)
@triton.jit
def decode_attention_kernel(
    q_ptr, k_ptr, v_ptr, output_ptr,
    scale,
    seq_k, head_dim,
    stride_q0, stride_q2,
    stride_k0, stride_k1, stride_k2,
    stride_v0, stride_v1, stride_v2,
    stride_o0, stride_o2,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """
    Single-query FlashAttention for decode steps (seq_q == 1).
    Grid: (batch * num_heads,)
    Reads Q once, streams K/V in BLOCK_K tiles with online softmax.
    """
    pid_bh = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    offs_k = tl.arange(0, BLOCK_K)

    # Load the single query vector
    q = tl.load(
        q_ptr + pid_bh * stride_q0 + offs_d * stride_q2,
        mask=offs_d < head_dim, other=0.0,
    )

    # Online softmax state
    m_i = tl.full([1], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([1], dtype=tl.float32)
    acc = tl.zeros([BLOCK_D], dtype=tl.float32)

    n_tiles = tl.cdiv(seq_k, BLOCK_K)
    for tile in range(n_tiles):
        k_start = tile * BLOCK_K
        cur_k   = k_start + offs_k

        k = tl.load(
            k_ptr + pid_bh * stride_k0 + cur_k[:, None] * stride_k1 + offs_d[None, :] * stride_k2,
            mask=(cur_k[:, None] < seq_k) & (offs_d[None, :] < head_dim), other=0.0,
        )
        score = tl.sum(k * q[None, :], axis=1) * scale
        score = tl.where(cur_k < seq_k, score, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(score, axis=0))
        alpha = tl.exp(m_i - m_new)
        exp_s = tl.exp(score - m_new)
        l_new = alpha * l_i + tl.sum(exp_s, axis=0)

        v = tl.load(
            v_ptr + pid_bh * stride_v0 + cur_k[:, None] * stride_v1 + offs_d[None, :] * stride_v2,
            mask=(cur_k[:, None] < seq_k) & (offs_d[None, :] < head_dim), other=0.0,
        )
        acc = alpha * acc + tl.sum(exp_s[:, None] * v, axis=0)

        m_i = m_new
        l_i = l_new

    out = acc / l_i
    tl.store(
        output_ptr + pid_bh * stride_o0 + offs_d * stride_o2,
        out, mask=offs_d < head_dim,
    )


# ============================================================================
# Optimization 1+2: FlashAttention-2 fused kernel with causal early-exit
#
# Causal early-exit uses a variable loop bound n_k_tiles computed from pid_q:
#   n_k_tiles = ceil((pid_q+1)*BLOCK_Q / BLOCK_K)   [clamped to total tiles]
# This avoids range() over tiles that would be entirely -inf, saving ~50%
# of K/V HBM loads during causal prefill.  Uses range(n_k_tiles) instead of
# break-inside-loop to stay compatible with all Triton versions.
# ============================================================================

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_Q": 16,  "BLOCK_K": 64},  num_warps=2, num_stages=3),
        triton.Config({"BLOCK_Q": 64,  "BLOCK_K": 64},  num_warps=4, num_stages=3),
        triton.Config({"BLOCK_Q": 64,  "BLOCK_K": 128}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_Q": 128, "BLOCK_K": 64},  num_warps=4, num_stages=4),
        triton.Config({"BLOCK_Q": 128, "BLOCK_K": 128}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_Q": 64,  "BLOCK_K": 64},  num_warps=4, num_stages=5),
        triton.Config({"BLOCK_Q": 128, "BLOCK_K": 64},  num_warps=8, num_stages=5),
    ],
    key=["seq_q", "seq_k"],
)
@triton.jit
def attention_fused_kernel(
    q_ptr, k_ptr, v_ptr, output_ptr, mask_ptr,
    scale,
    seq_q, seq_k, head_dim,
    stride_q0, stride_q1, stride_q2,
    stride_k0, stride_k1, stride_k2,
    stride_v0, stride_v1, stride_v2,
    stride_o0, stride_o1, stride_o2,
    stride_m0, stride_m1, stride_m2,
    IS_CAUSAL: tl.constexpr,
    HAS_MASK:  tl.constexpr,
    BLOCK_Q:   tl.constexpr,
    BLOCK_K:   tl.constexpr,
    BLOCK_D:   tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_q  = tl.program_id(1)

    offs_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    offs_d = tl.arange(0, BLOCK_D)
    offs_k = tl.arange(0, BLOCK_K)

    q = tl.load(
        q_ptr + pid_bh * stride_q0 + offs_q[:, None] * stride_q1 + offs_d[None, :] * stride_q2,
        mask=(offs_q[:, None] < seq_q) & (offs_d[None, :] < head_dim), other=0.0,
    )

    m_prev = tl.zeros([BLOCK_Q], dtype=tl.float32) - float("inf")
    l_prev = tl.zeros([BLOCK_Q], dtype=tl.float32)
    acc    = tl.zeros([BLOCK_Q, BLOCK_D], dtype=tl.float32)

    # Causal early-exit: limit K tile iterations to those that can have
    # at least one valid (non-masked) entry for this Q block.
    total_k_tiles = tl.cdiv(seq_k, BLOCK_K)
    if IS_CAUSAL:
        causal_limit = tl.cdiv((pid_q + 1) * BLOCK_Q, BLOCK_K)
        n_k_tiles    = tl.minimum(causal_limit, total_k_tiles)
    else:
        n_k_tiles = total_k_tiles

    for tile_idx in range(n_k_tiles):
        start_k = tile_idx * BLOCK_K
        curr_k  = start_k + offs_k

        # K transposed: (head_dim, BLOCK_K)
        k = tl.load(
            k_ptr + pid_bh * stride_k0 + curr_k[None, :] * stride_k1 + offs_d[:, None] * stride_k2,
            mask=(curr_k[None, :] < seq_k) & (offs_d[:, None] < head_dim), other=0.0,
        )
        v = tl.load(
            v_ptr + pid_bh * stride_v0 + curr_k[:, None] * stride_v1 + offs_d[None, :] * stride_v2,
            mask=(curr_k[:, None] < seq_k) & (offs_d[None, :] < head_dim), other=0.0,
        )

        qk = tl.dot(q, k, allow_tf32=True) * scale

        if HAS_MASK:
            m_ptrs = mask_ptr + pid_bh * stride_m0 + offs_q[:, None] * stride_m1 + curr_k[None, :] * stride_m2
            m_mask = tl.load(m_ptrs,
                mask=(offs_q[:, None] < seq_q) & (curr_k[None, :] < seq_k), other=float("-inf"))
            qk = qk + m_mask

        if IS_CAUSAL:
            qk = tl.where(offs_q[:, None] >= curr_k[None, :], qk, float("-inf"))

        qk = tl.where(curr_k[None, :] < seq_k, qk, float("-inf"))
        qk = tl.where(offs_q[:, None]  < seq_q, qk, float("-inf"))

        m_curr = tl.max(qk, axis=1)
        m_new  = tl.maximum(m_prev, m_curr)
        p      = tl.exp(qk - m_new[:, None])
        alpha  = tl.exp(m_prev - m_new)
        l_curr = tl.sum(p, axis=1)
        l_new  = alpha * l_prev + l_curr

        acc    = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v, allow_tf32=True)

        m_prev = m_new
        l_prev = l_new

    output = acc / l_prev[:, None]
    tl.store(
        output_ptr + pid_bh * stride_o0 + offs_q[:, None] * stride_o1 + offs_d[None, :] * stride_o2,
        output,
        mask=(offs_q[:, None] < seq_q) & (offs_d[None, :] < head_dim),
    )


# ============================================================================
# Attention Classes
# ============================================================================

class MultiHeadAttention:
    """Multi-head attention using Triton kernels."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
    ):
        self.hidden_size  = hidden_size
        self.num_heads    = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim     = head_dim or (hidden_size // num_heads)
        self.scale        = 1.0 / np.sqrt(self.head_dim)
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        is_causal: bool = False,
    ) -> torch.Tensor:
        batch, num_heads, seq_q, head_dim = q.shape
        _, num_kv_heads, seq_k, _         = k.shape
        if num_kv_heads != num_heads:
            k = self._expand_kv(k, self.num_queries_per_kv)
            v = self._expand_kv(v, self.num_queries_per_kv)
        return scaled_dot_product_attention(q, k, v, attention_mask, is_causal, self.scale)

    def _expand_kv(self, x: torch.Tensor, num_repeats: int) -> torch.Tensor:
        batch, num_kv_heads, seq_len, head_dim = x.shape
        x_expanded = x[:, :, None, :, :].expand(batch, num_kv_heads, num_repeats, seq_len, head_dim)
        return x_expanded.reshape(batch, num_kv_heads * num_repeats, seq_len, head_dim)


def next_power_of_two(x: int) -> int:
    return 1 << (x - 1).bit_length() if x > 0 else 1


MAX_ATTENTION_DIM = 256


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    is_causal: bool = False,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Scaled dot-product attention dispatch:
      seq_q == 1 and no mask  -> decode_attention_kernel (Opt 3)
      seq_q > 1               -> attention_fused_kernel  (Opt 1+2, FA-2 + causal early-exit)
      not CUDA / dim too large -> PyTorch fallback
    """
    batch, num_heads, seq_q, head_dim = q.shape
    _,     _,         seq_k, _        = k.shape

    if scale is None:
        scale = 1.0 / np.sqrt(head_dim)

    head_dim_padded = next_power_of_two(head_dim)
    use_triton = q.is_cuda and head_dim_padded <= MAX_ATTENTION_DIM

    if use_triton:
        q_flat = q.reshape(batch * num_heads, seq_q, head_dim).to(torch.float32).contiguous()
        k_flat = k.reshape(batch * num_heads, seq_k, head_dim).to(torch.float32).contiguous()
        v_flat = v.reshape(batch * num_heads, seq_k, head_dim).to(torch.float32).contiguous()

        def _pad_hd(t, sq):
            p = torch.zeros(t.shape[0], sq, head_dim_padded, dtype=t.dtype, device=t.device)
            p[:, :, :head_dim] = t
            return p

        if head_dim_padded != head_dim:
            q_flat = _pad_hd(q_flat, seq_q)
            k_flat = _pad_hd(k_flat, seq_k)
            v_flat = _pad_hd(v_flat, seq_k)

        # ── Decode path (seq_q == 1, no additive mask) ───────────────────
        if seq_q == 1 and attention_mask is None:
            q_2d  = q_flat.reshape(batch * num_heads, head_dim_padded)
            out_2d = torch.empty_like(q_2d)

            decode_attention_kernel[(batch * num_heads,)](
                q_2d, k_flat, v_flat, out_2d,
                float(scale),
                seq_k, head_dim_padded,
                q_2d.stride(0), q_2d.stride(1),
                k_flat.stride(0), k_flat.stride(1), k_flat.stride(2),
                v_flat.stride(0), v_flat.stride(1), v_flat.stride(2),
                out_2d.stride(0), out_2d.stride(1),
            )
            out = out_2d[:, :head_dim]
            return out.reshape(batch, num_heads, 1, head_dim).to(q.dtype)

        # ── Prefill path (seq_q > 1): FA-2 with causal early-exit ────────
        output  = torch.empty(
            (batch * num_heads, seq_q, head_dim_padded),
            dtype=torch.float32, device=q.device,
        )
        has_mask = attention_mask is not None

        if has_mask:
            attn_mask = attention_mask
            if attn_mask.ndim == 4:
                attn_mask = attn_mask.reshape(batch * num_heads, seq_q, seq_k)
            attn_mask = attn_mask.to(torch.float32).contiguous()
            s_m0, s_m1, s_m2 = attn_mask.stride()
            mask_ptr = attn_mask
        else:
            s_m0 = s_m1 = s_m2 = 0
            mask_ptr = q_flat

        grid = lambda META: (batch * num_heads, triton.cdiv(seq_q, META["BLOCK_Q"]))

        attention_fused_kernel[grid](
            q_flat, k_flat, v_flat, output, mask_ptr,
            float(scale),
            seq_q, seq_k, head_dim_padded,
            q_flat.stride(0), q_flat.stride(1), q_flat.stride(2),
            k_flat.stride(0), k_flat.stride(1), k_flat.stride(2),
            v_flat.stride(0), v_flat.stride(1), v_flat.stride(2),
            output.stride(0), output.stride(1), output.stride(2),
            s_m0, s_m1, s_m2,
            IS_CAUSAL=is_causal,
            HAS_MASK=has_mask,
            BLOCK_D=head_dim_padded,
        )

        out = output[:, :, :head_dim]
        return out.reshape(batch, num_heads, seq_q, head_dim).to(q.dtype)

    # ── PyTorch fallback ─────────────────────────────────────────────────
    scores = torch.einsum("bnqd,bnkd->bnqk", q, k) * scale
    if is_causal:
        mask = torch.triu(
            torch.ones((seq_q, seq_k), dtype=torch.float32, device=q.device), diagonal=1
        ) * -1e9
        scores = scores + mask[None, None, :, :]
    if attention_mask is not None:
        scores = scores + attention_mask
    scores = scores - torch.max(scores, dim=-1, keepdim=True).values
    attn_w = torch.exp(scores)
    attn_w = attn_w / torch.sum(attn_w, dim=-1, keepdim=True)
    output = torch.einsum("bnqk,bnkd->bnqd", attn_w, v)
    return output.to(q.dtype)


if __name__ == "__main__":
    print("Testing Triton Attention...")
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    num_heads  = 4
    seq_len    = 16
    head_dim   = 64

    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

    print("\nBasic attention:")
    output = scaled_dot_product_attention(q, k, v)
    print(f"  Output shape: {output.shape}")

    print("\nCausal attention:")
    output_causal = scaled_dot_product_attention(q, k, v, is_causal=True)
    print(f"  Output shape: {output_causal.shape}")

    print("\nDecode-optimised (seq_q=1):")
    q1    = torch.randn(batch_size, num_heads, 1, head_dim, device=device)
    out_d = scaled_dot_product_attention(q1, k, v)
    print(f"  Output shape: {out_d.shape}")

    print("\nWith attention mask:")
    mask = torch.zeros((batch_size, num_heads, seq_len, seq_len),
                       dtype=torch.float32, device=device)
    mask[:, :, :, seq_len // 2:] = -1e9
    output_masked = scaled_dot_product_attention(q, k, v, attention_mask=mask)
    print(f"  Output shape: {output_masked.shape}")

    print("\nGrouped Query Attention (GQA):")
    num_kv_heads = 2
    k_gqa = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device=device)
    v_gqa = torch.randn(batch_size, num_kv_heads, seq_len, head_dim, device=device)
    attn  = MultiHeadAttention(
        hidden_size=num_heads * head_dim, num_heads=num_heads, num_kv_heads=num_kv_heads,
    )
    output_gqa = attn(q, k_gqa, v_gqa)
    print(f"  Output shape: {output_gqa.shape}")

    print("\nOutput statistics:")
    print(f"  Mean: {float(output.mean()):.4f}")
    print(f"  Std:  {float(output.std()):.4f}")
    print(f"  Min:  {float(output.min()):.4f}")
    print(f"  Max:  {float(output.max()):.4f}")

    print("\nTriton Attention working!")
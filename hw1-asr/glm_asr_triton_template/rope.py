"""
Triton Rotary Position Embeddings (RoPE)
End-to-end implementation using Triton kernels

*** STUDENT ASSIGNMENT ***
Fill in the TODO sections to implement RoPE using Triton kernels
"""

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl


def get_stream():
    """Get current CUDA stream pointer."""
    if torch.cuda.is_available():
        return torch.cuda.current_stream().cuda_stream
    return None


# ============================================================================
# Triton Kernels for RoPE
# ============================================================================

@triton.jit
def compute_freqs_kernel(
    positions_ptr,
    inv_freq_ptr,
    cos_ptr,
    sin_ptr,
    seq_len,
    half_dim,
    stride_pos,
    stride_inv,
    stride_cos0,
    stride_cos1,
    stride_sin0,
    stride_sin1,
    BLOCK: tl.constexpr,
):
    """
    Compute cos and sin for rotary embeddings.

    *** TODO: Implement this kernel ***

    Grid: (seq_len,)
    """
    pid = tl.program_id(0)

    # ============================================================================
    # TODO: Implement frequency computation
    # ============================================================================
    #
    # Step 1: Load position as scalar
    # Step 2: Load inverse frequencies
    # Step 3: Compute freqs = position * inv_freq
    # Step 4: Compute cos and sin
    # Step 5: Store concatenated cos/sin
    offs = tl.arange(0, BLOCK)
    mask = offs < half_dim

    pos = tl.load(positions_ptr + pid * stride_pos)
    inv = tl.load(inv_freq_ptr + offs * stride_inv, mask=mask, other=0.0)
    freqs = pos * inv

    cos_half = tl.cos(freqs)
    sin_half = tl.sin(freqs)

    # Store cos: [cos, cos] (duplicated across both halves)
    tl.store(cos_ptr + pid * stride_cos0 + offs * stride_cos1, cos_half, mask=mask)
    tl.store(
        cos_ptr + pid * stride_cos0 + (offs + half_dim) * stride_cos1,
        cos_half,
        mask=mask,
    )
    # Store sin: [sin, sin]
    tl.store(sin_ptr + pid * stride_sin0 + offs * stride_sin1, sin_half, mask=mask)
    tl.store(
        sin_ptr + pid * stride_sin0 + (offs + half_dim) * stride_sin1,
        sin_half,
        mask=mask,
    )


# ============================================================================
# RoPE Classes
# ============================================================================

class RotaryEmbedding:
    """Rotary Position Embedding using Triton."""

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 8192,
        base: float = 10000.0,
        partial_rotary_factor: float = 1.0,
    ):
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.partial_rotary_factor = partial_rotary_factor

        self.rotary_dim = int(dim * partial_rotary_factor)
        self.rotary_dim = self.rotary_dim - (self.rotary_dim % 2)

        inv_freq = 1.0 / (
            base ** (torch.arange(0, self.rotary_dim, 2, dtype=torch.float32) / self.rotary_dim)
        )
        self.inv_freq = inv_freq

        self._update_cache(max_position_embeddings)

    def _update_cache(self, seq_len: int, device: Optional[torch.device] = None):
        """Pre-compute cos and sin using Triton kernel."""
        self.max_seq_len_cached = seq_len
        half_dim = self.rotary_dim // 2
        if device is None:
            device = self.inv_freq.device

        positions = torch.arange(seq_len, dtype=torch.float32, device=device)
        cos_cache = torch.empty((seq_len, self.rotary_dim), dtype=torch.float32, device=device)
        sin_cache = torch.empty((seq_len, self.rotary_dim), dtype=torch.float32, device=device)

        if device.type == "cuda":
            if self.inv_freq.device != device:
                self.inv_freq = self.inv_freq.to(device)

            block = triton.next_power_of_2(half_dim)
            compute_freqs_kernel[(seq_len,)](
                positions,
                self.inv_freq,
                cos_cache,
                sin_cache,
                seq_len,
                half_dim,
                positions.stride(0),
                self.inv_freq.stride(0),
                cos_cache.stride(0),
                cos_cache.stride(1),
                sin_cache.stride(0),
                sin_cache.stride(1),
                BLOCK=block,
            )
        else:
            if self.inv_freq.device != device:
                self.inv_freq = self.inv_freq.to(device)
            freqs = positions[:, None] * self.inv_freq[None, :]
            cos_half = torch.cos(freqs)
            sin_half = torch.sin(freqs)
            cos_cache[:, :half_dim] = cos_half
            cos_cache[:, half_dim : half_dim * 2] = cos_half
            sin_cache[:, :half_dim] = sin_half
            sin_cache[:, half_dim : half_dim * 2] = sin_half

        self.cos_cached = cos_cache
        self.sin_cached = sin_cache

    def __call__(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get cos and sin for given positions."""
        seq_len = x.shape[-2]

        if seq_len > self.max_seq_len_cached:
            self._update_cache(seq_len, device=x.device)
        elif self.cos_cached.device != x.device:
            self._update_cache(self.max_seq_len_cached, device=x.device)

        if position_ids is not None:
            cos = self.cos_cached[position_ids].to(x.dtype)
            sin = self.sin_cached[position_ids].to(x.dtype)
            if cos.ndim == 3 and cos.shape[0] == 1:
                cos = cos[0]
                sin = sin[0]
        else:
            cos = self.cos_cached[:seq_len].to(x.dtype)
            sin = self.sin_cached[:seq_len].to(x.dtype)

        return cos, sin


def next_power_of_two(x: int) -> int:
    """Return the smallest power of two >= x."""
    return 1 << (x - 1).bit_length() if x > 0 else 1


MAX_ROPE_DIM = 256


def _apply_rope_single(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    half_dim: int,
    head_dim: int,
) -> torch.Tensor:
    """Apply RoPE to a single tensor (Q or K) using Torch."""
    batch, num_heads, seq_len, _ = x.shape

    cos = cos[:seq_len]
    sin = sin[:seq_len]

    x1 = x[..., :half_dim]
    x2 = x[..., half_dim : half_dim * 2]

    cos_expanded = cos[None, None, :, :]
    sin_expanded = sin[None, None, :, :]

    x1_rot = x1 * cos_expanded - x2 * sin_expanded
    x2_rot = x2 * cos_expanded + x1 * sin_expanded

    if head_dim > half_dim * 2:
        x_pass = x[..., half_dim * 2 :]
        return torch.cat([x1_rot, x2_rot, x_pass], dim=-1)
    return torch.cat([x1_rot, x2_rot], dim=-1)


@triton.jit
def apply_rope_kernel(
    x_ptr,
    cos_ptr,
    sin_ptr,
    out_ptr,
    seq_m, # batch * num_heads
    seq_len,
    head_dim,
    rotary_dim,
    stride_x0, stride_x1, stride_x2, stride_x3,
    stride_cos0, stride_cos1,
    stride_sin0, stride_sin1,
    stride_out0, stride_out1, stride_out2, stride_out3,
    BLOCK_D: tl.constexpr,
):
    """
    Apply RoPE to a tensor.
    Grid: (batch * num_heads, seq_len)
    """
    pid_bh = tl.program_id(0)
    pid_s = tl.program_id(1)
    
    half_dim = rotary_dim // 2
    offs_d = tl.arange(0, BLOCK_D)
    
    # Load x
    x_ptrs = x_ptr + pid_bh * stride_x0 + pid_s * stride_x2 + offs_d * stride_x3
    x = tl.load(x_ptrs, mask=offs_d < head_dim, other=0.0)
    
    # Load cos/sin (only for rotary_dim)
    # Note: offs_d % rotary_dim handles duplicated cos/sin in cache
    cos = tl.load(cos_ptr + pid_s * stride_cos0 + (offs_d % rotary_dim) * stride_cos1, mask=offs_d < rotary_dim, other=1.0)
    sin = tl.load(sin_ptr + pid_s * stride_sin0 + (offs_d % rotary_dim) * stride_sin1, mask=offs_d < rotary_dim, other=0.0)
    
    # For RoPE rotation:
    # x_other: for first half, it's x2. For second half, it's x1.
    other_offs = tl.where(offs_d < half_dim, offs_d + half_dim, offs_d - half_dim)
    x_other_ptrs = x_ptr + pid_bh * stride_x0 + pid_s * stride_x2 + other_offs * stride_x3
    x_other = tl.load(x_other_ptrs, mask=offs_d < rotary_dim, other=0.0)
    
    # out = x * cos + x_rotated * sin
    # x_rotated = [-x2, x1]
    sign = tl.where(offs_d < half_dim, -1.0, 1.0)
    out = tl.where(offs_d < rotary_dim, x * cos + (x_other * sin * sign), x)
    
    # Store
    out_ptrs = out_ptr + pid_bh * stride_out0 + pid_s * stride_out2 + offs_d * stride_out3
    tl.store(out_ptrs, out, mask=offs_d < head_dim)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary position embeddings using Triton if available.
    """
    batch, num_q_heads, seq_len, head_dim = q.shape
    _, num_kv_heads, _, _ = k.shape

    # Ensure rotary_dim and head_dim are integers
    head_dim = int(head_dim)
    if rotary_dim is None:
        rotary_dim = head_dim
    else:
        rotary_dim = int(rotary_dim)

    if not q.is_cuda:
        half_dim = rotary_dim // 2
        
        # Crop cos/sin to match half_dim (Torch fallback expectation)
        if cos.shape[1] > half_dim:
            cos_crop = cos[:seq_len, :half_dim]
            sin_crop = sin[:seq_len, :half_dim]
        else:
            cos_crop, sin_crop = cos, sin

        q_out = _apply_rope_single(q, cos_crop, sin_crop, half_dim, head_dim)
        k_out = _apply_rope_single(k, cos_crop, sin_crop, half_dim, head_dim)

        return q_out.to(q.dtype), k_out.to(k.dtype)

    # Triton path
    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    
    block_d = next_power_of_two(head_dim)
    
    # Apply to Q
    apply_rope_kernel[(batch * num_q_heads, seq_len)](
        q, cos, sin, q_out,
        batch * num_q_heads, seq_len, head_dim, rotary_dim,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        cos.stride(0), cos.stride(1),
        sin.stride(0), sin.stride(1),
        q_out.stride(0), q_out.stride(1), q_out.stride(2), q_out.stride(3),
        BLOCK_D=block_d,
    )
    
    # Apply to K
    apply_rope_kernel[(batch * num_kv_heads, seq_len)](
        k, cos, sin, k_out,
        batch * num_kv_heads, seq_len, head_dim, rotary_dim,
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        cos.stride(0), cos.stride(1),
        sin.stride(0), sin.stride(1),
        k_out.stride(0), k_out.stride(1), k_out.stride(2), k_out.stride(3),
        BLOCK_D=block_d,
    )

    return q_out, k_out


def apply_partial_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rotary_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply rotary embeddings to partial dimensions."""
    return apply_rotary_pos_emb(q, k, cos, sin, rotary_dim)


if __name__ == "__main__":
    print("Testing Triton RoPE...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 2
    num_heads = 4
    seq_len = 16
    head_dim = 64

    rope = RotaryEmbedding(dim=head_dim, max_position_embeddings=1024)

    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)

    cos, sin = rope(q)
    print(f"Cos shape: {cos.shape}")
    print(f"Sin shape: {sin.shape}")

    q_rot, k_rot = apply_rotary_pos_emb(q, k, cos, sin)
    print(f"Q rotated shape: {q_rot.shape}")
    print(f"K rotated shape: {k_rot.shape}")

    print("\nTesting partial RoPE (50%):")
    rope_partial = RotaryEmbedding(dim=head_dim, partial_rotary_factor=0.5)
    cos_p, sin_p = rope_partial(q)
    q_rot_p, k_rot_p = apply_partial_rotary_pos_emb(q, k, cos_p, sin_p, head_dim // 2)
    print(f"Q rotated (partial) shape: {q_rot_p.shape}")

    print("\nTriton RoPE working!")

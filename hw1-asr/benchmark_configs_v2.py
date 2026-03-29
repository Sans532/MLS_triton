#!/usr/bin/env python3
"""
benchmark_configs_v2.py — Benchmarks each tile config individually.
Defines a bare (no-autotune) copy of the kernel and calls it directly.

Run from hw1-asr/:
    python benchmark_configs_v2.py glm_asr_triton_template
"""
import sys, os, time
import torch
import triton
import triton.language as tl
import numpy as np

folder = sys.argv[1] if len(sys.argv) > 1 else "glm_asr_triton_template"
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, folder))
for mod in list(sys.modules):
    if mod in ["layers","attention","rope","model","weight_loader"]:
        del sys.modules[mod]

device = torch.device("cuda")
WARMUP = 3
RUNS   = 10

# ── Bare kernel (no autotune) — same logic as linear_kernel_tf32 ─────────────
@triton.jit
def _linear_bare(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptr + offs_m[:,None]*stride_am + (k+offs_k[None,:])*stride_ak,
                    mask=(offs_m[:,None]<M) & (k+offs_k[None,:]<K), other=0.0)
        b = tl.load(b_ptr + (k+offs_k[:,None])*stride_bk + offs_n[None,:]*stride_bn,
                    mask=(k+offs_k[:,None]<K) & (offs_n[None,:]<N), other=0.0)
        acc += tl.dot(a, b, allow_tf32=True)
    tl.store(c_ptr + offs_m[:,None]*stride_cm + offs_n[None,:]*stride_cn,
             acc, mask=(offs_m[:,None]<M) & (offs_n[None,:]<N))


# ── Bare attention kernel (no autotune) ───────────────────────────────────────
@triton.jit
def _attn_bare(
    q_ptr, k_ptr, v_ptr, out_ptr,
    stride_qb, stride_qh, stride_qq, stride_qd,
    stride_kb, stride_kh, stride_kk, stride_kd,
    stride_vb, stride_vh, stride_vk, stride_vd,
    stride_ob, stride_oh, stride_oq, stride_od,
    seq_q, seq_k, scale,
    IS_CAUSAL: tl.constexpr,
    BLOCK_Q:   tl.constexpr,
    BLOCK_K:   tl.constexpr,
    HEAD_DIM:  tl.constexpr,
):
    pid_q   = tl.program_id(0)
    pid_h   = tl.program_id(1)
    pid_b   = tl.program_id(2)

    q_off = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
    d_off = tl.arange(0, HEAD_DIM)

    q = tl.load(q_ptr + pid_b*stride_qb + pid_h*stride_qh +
                q_off[:,None]*stride_qq + d_off[None,:]*stride_qd,
                mask=q_off[:,None] < seq_q, other=0.0)

    m_i   = tl.full([BLOCK_Q], float("-inf"), dtype=tl.float32)
    l_i   = tl.zeros([BLOCK_Q], dtype=tl.float32)
    acc   = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)

    for k_start in range(0, seq_k, BLOCK_K):
        k_off = k_start + tl.arange(0, BLOCK_K)
        k = tl.load(k_ptr + pid_b*stride_kb + pid_h*stride_kh +
                    k_off[:,None]*stride_kk + d_off[None,:]*stride_kd,
                    mask=k_off[:,None] < seq_k, other=0.0)
        v = tl.load(v_ptr + pid_b*stride_vb + pid_h*stride_vh +
                    k_off[:,None]*stride_vk + d_off[None,:]*stride_vd,
                    mask=k_off[:,None] < seq_k, other=0.0)

        scores = tl.dot(q, tl.trans(k)) * scale
        if IS_CAUSAL:
            causal_mask = q_off[:,None] >= k_off[None,:]
            scores = tl.where(causal_mask, scores, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha  = tl.exp(m_i - m_new)
        scores = tl.exp(scores - m_new[:,None])
        l_i    = alpha * l_i + tl.sum(scores, axis=1)
        acc    = alpha[:,None] * acc + tl.dot(scores, v)
        m_i    = m_new

    acc = acc / l_i[:,None]
    tl.store(out_ptr + pid_b*stride_ob + pid_h*stride_oh +
             q_off[:,None]*stride_oq + d_off[None,:]*stride_od,
             acc, mask=q_off[:,None] < seq_q)


# ── Timing helper ─────────────────────────────────────────────────────────────
def bench(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(RUNS):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times)), float(np.std(times))


# ── Linear benchmark ──────────────────────────────────────────────────────────
linear_configs = [
    (16,  16,  64,  2, "C1  16x16x64 "),
    (16,  32,  128, 2, "C2  16x32x128"),
    (16,  64,  64,  2, "C3  16x64x64 "),
    (32,  32,  64,  2, "C9  32x32x64 "),
    (32,  64,  64,  2, "C10 32x64x64 "),
    (64,  32,  64,  2, "C11 64x32x64 "),
    (32,  128, 32,  4, "C12 32x128x32"),
    (128, 32,  32,  4, "C13 128x32x32"),
    (64,  64,  64,  4, "C7  64x64x64 "),
    (128, 64,  32,  4, "C4  128x64x32"),
    (64,  128, 32,  4, "C5  64x128x32"),
    (128, 128, 32,  8, "C6  128x128x32"),
]

shapes = [
    ("Audio enc Q (M=750,K=1280,N=1280) ", 750,  1280, 1280),
    ("Audio enc fc1(M=750,K=1280,N=5120)", 750,  1280, 5120),
    ("Dec Q prefill(M=222,K=3584,N=3584)", 222,  3584, 3584),
    ("Dec Q decode (M=1,  K=3584,N=3584)", 1,    3584, 3584),
    ("Dec MLP gate (M=222,K=3584,N=18944)", 222, 3584, 18944),
]

W = 38
print("=" * 85)
print("LINEAR KERNEL — ISOLATED CONFIG BENCHMARK (bare kernel, no autotune wrapper)")
print("=" * 85)
print(f"  {'Shape':<{W}} {'Config':<16} {'BM':>4}{'BN':>4}{'BK':>4}{'nw':>3}  {'ms':>8}  {'std':>7}")
print("-" * 85)

for shape_name, M, K, N in shapes:
    A = torch.randn(M, K, device=device, dtype=torch.float32)
    B = torch.randn(N, K, device=device, dtype=torch.float32).t().contiguous()
    results = []
    for BM, BN, BK, nw, label in linear_configs:
        C = torch.empty(M, N, device=device, dtype=torch.float32)
        grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
        try:
            fn = lambda bm=BM, bn=BN, bk=BK, nwarps=nw: \
                _linear_bare[grid](
                    A, B, C, M, N, K,
                    A.stride(0), A.stride(1),
                    B.stride(0), B.stride(1),
                    C.stride(0), C.stride(1),
                    BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
                    num_warps=nwarps
                )
            ms, std = bench(fn)
            results.append((ms, std, label, BM, BN, BK, nw))
        except Exception as e:
            results.append((float('inf'), 0, label, BM, BN, BK, nw))

    results.sort(key=lambda x: x[0])
    for i, (ms, std, label, BM, BN, BK, nw) in enumerate(results):
        tag = "  ← WINNER" if i == 0 else ""
        ms_str = f"{ms:>8.3f}" if ms < 900 else "   ERROR"
        print(f"  {shape_name:<{W}} {label:<16} {BM:>4}{BN:>4}{BK:>4}{nw:>3}  {ms_str}ms  {std:>6.3f}{tag}")
    print()


# ── Attention benchmark ───────────────────────────────────────────────────────
attn_configs = [
    (16,  64,  2, "C1  BQ=16,BK=64 "),
    (64,  64,  4, "C2  BQ=64,BK=64 "),
    (64,  128, 4, "C3  BQ=64,BK=128"),
    (128, 64,  4, "C4  BQ=128,BK=64"),
    (128, 128, 8, "C5  BQ=128,BK=128"),
    (32,  64,  2, "C8  BQ=32,BK=64 "),
]

attn_shapes = [
    ("Audio encoder (sq=750,sk=750,h=20,d=64)  ", 750, 750,  20, 64, False),
    ("Dec prefill   (sq=222,sk=222,h=28,d=128) ", 222, 222,  28, 128, True),
    ("Dec decode    (sq=1,  sk=222,h=28,d=128) ",   1, 222,  28, 128, True),
]

print("=" * 85)
print("ATTENTION KERNEL — ISOLATED CONFIG BENCHMARK")
print("=" * 85)
print(f"  {'Shape':<{W+4}} {'Config':<18} {'BQ':>4}{'BK':>4}{'nw':>3}  {'ms':>8}  {'std':>7}")
print("-" * 85)

for shape_name, sq, sk, nh, hd, causal in attn_shapes:
    Q   = torch.randn(1, nh, sq, hd, device=device, dtype=torch.float32)
    K   = torch.randn(1, nh, sk, hd, device=device, dtype=torch.float32)
    V   = torch.randn(1, nh, sk, hd, device=device, dtype=torch.float32)
    OUT = torch.empty_like(Q)
    scale = hd ** -0.5
    results = []

    for BQ, BK, nw, label in attn_configs:
        if BQ > sq and sq > 0:
            BQ = max(16, sq)   # clamp for small seq
        grid = (triton.cdiv(sq, BQ), nh, 1)
        try:
            fn = lambda bq=BQ, bk=BK, nwarps=nw, c=causal: \
                _attn_bare[grid](
                    Q, K, V, OUT,
                    Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
                    K.stride(0), K.stride(1), K.stride(2), K.stride(3),
                    V.stride(0), V.stride(1), V.stride(2), V.stride(3),
                    OUT.stride(0), OUT.stride(1), OUT.stride(2), OUT.stride(3),
                    sq, sk, scale,
                    IS_CAUSAL=c,
                    BLOCK_Q=bq, BLOCK_K=bk,
                    HEAD_DIM=hd,
                    num_warps=nwarps
                )
            ms, std = bench(fn)
            results.append((ms, std, label, BQ, BK, nw))
        except Exception as e:
            results.append((float('inf'), 0, label, BQ, BK, nw))

    results.sort(key=lambda x: x[0])
    for i, (ms, std, label, BQ, BK, nw) in enumerate(results):
        tag = "  ← WINNER" if i == 0 else ""
        ms_str = f"{ms:>8.4f}" if ms < 900 else "   ERROR"
        print(f"  {shape_name:<{W+4}} {label:<18} {BQ:>4}{BK:>4}{nw:>3}  {ms_str}ms  {std:>6.4f}{tag}")
    print()

print("Done.")

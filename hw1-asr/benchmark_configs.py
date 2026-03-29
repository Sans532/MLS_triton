"""
Micro-benchmark each autotune config individually for the key shapes.
Run from hw1-asr/ on the cluster.

Usage:
    python benchmark_configs.py glm_asr_triton_template

This runs each config 10 times and reports mean latency per config,
showing exactly why autotune picks the winner it does.
"""
import sys, os, time
import torch
import triton
import numpy as np

folder = sys.argv[1] if len(sys.argv) > 1 else "glm_asr_triton_template"
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, folder))

for mod in list(sys.modules):
    if mod in ["layers","attention","rope","model","weight_loader"]:
        del sys.modules[mod]

import layers
import attention as attn_mod

device = torch.device("cuda")
WARMUP = 5
RUNS   = 20

def bench(fn, warmup=WARMUP, runs=RUNS):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.mean(times)), float(np.std(times))

def run_linear_config(M, K, N, BM, BN, BK, nw, ns):
    A = torch.randn(M, K, device=device, dtype=torch.float32)
    B = torch.randn(K, N, device=device, dtype=torch.float32)
    C = torch.empty(M, N, device=device, dtype=torch.float32)
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    fn = lambda: layers.linear_kernel_tf32[grid](
        A, B, C, M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK,
        num_warps=nw, num_stages=ns
    )
    return bench(fn)

linear_configs = [
    # (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages, label)
    # accum = BM×BN×4 bytes lives in registers
    # shmem = stages×(BM×BK + BK×BN)×4 bytes — keep < 96KB to avoid Triton error
    (16,  16,  64,  2, 3, "C1  16x16x64 "),   # accum=1KB,  shmem=24KB  tiny
    (16,  32,  128, 2, 3, "C2  16x32x128"),   # accum=2KB,  shmem=72KB
    (16,  64,  64,  2, 3, "C3  16x64x64 "),   # accum=4KB,  shmem=60KB
    (32,  32,  64,  2, 3, "C9  32x32x64 "),   # accum=4KB,  shmem=48KB  NEW
    (32,  64,  64,  2, 3, "C10 32x64x64 "),   # accum=8KB,  shmem=72KB  NEW
    (64,  32,  64,  2, 3, "C11 64x32x64 "),   # accum=8KB,  shmem=72KB  NEW
    (32,  128, 32,  4, 3, "C12 32x128x32"),   # accum=16KB, shmem=60KB  NEW
    (128, 32,  32,  4, 3, "C13 128x32x32"),   # accum=16KB, shmem=60KB  NEW
    (64,  64,  64,  4, 4, "C7  64x64x64s4"),  # accum=16KB, shmem=128KB works on H200
    (64,  64,  64,  4, 5, "C8  64x64x64s5"),  # accum=16KB, shmem=160KB works on H200
    (128, 64,  32,  4, 3, "C4  128x64x32"),   # accum=32KB, shmem=72KB
    (64,  128, 32,  4, 3, "C5  64x128x32"),   # accum=32KB, shmem=72KB
    (128, 128, 32,  8, 3, "C6  128x128x32"),  # accum=64KB, shmem=96KB  largest safe
]

shapes = [
    ("Audio encoder Q proj     ", 750,  1280, 1280),
    ("Audio encoder fc1        ", 750,  1280, 5120),
    ("Decoder Q proj (prefill) ", 222,  3584, 3584),
    ("Decoder Q proj (decode)  ",   1,  3584, 3584),
    ("Decoder MLP gate (prefill)", 222, 3584, 18944),
]

print("=" * 80)
print("LINEAR KERNEL CONFIG BENCHMARK")
print("=" * 80)
print(f"{'Shape':<30} {'Config':<10} {'BM':>4} {'BN':>4} {'BK':>4} {'nw':>3} {'ns':>3} {'ms':>8} {'std':>7}")
print("-" * 80)

for shape_name, M, K, N in shapes:
    results = []
    for BM, BN, BK, nw, ns, cname in linear_configs:
        try:
            ms, std = run_linear_config(M, K, N, BM, BN, BK, nw, ns)
            results.append((ms, std, cname, BM, BN, BK, nw, ns))
        except Exception as e:
            results.append((999, 0, cname, BM, BN, BK, nw, ns))

    results.sort(key=lambda x: x[0])
    for i, (ms, std, cname, BM, BN, BK, nw, ns) in enumerate(results):
        winner = " ← WINNER" if i == 0 else ""
        print(f"  {shape_name:<28} {cname:<10} {BM:>4} {BN:>4} {BK:>4} {nw:>3} {ns:>3} "
              f"{ms:>7.3f}ms {std:>6.3f}{winner}")
    print()

# ── Attention configs ─────────────────────────────────────────────────────────
print("=" * 80)
print("ATTENTION KERNEL CONFIG BENCHMARK")
print("=" * 80)

attn_configs = [
    (16,  64,  2, 3, "Config 1"),
    (64,  64,  4, 3, "Config 2"),
    (64,  128, 4, 4, "Config 3"),
    (128, 64,  4, 4, "Config 4"),
    (128, 128, 8, 3, "Config 5"),
    (64,  64,  4, 5, "Config 6"),
    (128, 64,  8, 5, "Config 7"),
]

def run_attn_config(seq_q, seq_k, n_heads, head_dim, BQ, BK, nw, ns, causal=False):
    Q = torch.randn(1, n_heads, seq_q, head_dim, device=device, dtype=torch.float32)
    K = torch.randn(1, n_heads, seq_k, head_dim, device=device, dtype=torch.float32)
    V = torch.randn(1, n_heads, seq_k, head_dim, device=device, dtype=torch.float32)
    scale = head_dim ** -0.5
    out   = torch.empty_like(Q)
    grid  = (triton.cdiv(seq_q, BQ), n_heads, 1)
    fn = lambda: attn_mod.attention_fused_kernel[grid](
        Q, K, V, out,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K.stride(0), K.stride(1), K.stride(2), K.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        seq_q, seq_k, scale,
        IS_CAUSAL=causal, HAS_MASK=False,
        BLOCK_Q=BQ, BLOCK_K=BK,
        num_warps=nw, num_stages=ns
    )
    return bench(fn)

attn_shapes = [
    ("Audio encoder (seq=750) ", 750, 750,  20, 64, False),
    ("Decoder prefill (seq=222)", 222, 222, 28, 128, True),
    ("Decoder decode (seq_q=1) ",   1, 222, 28, 128, True),
]

print(f"{'Shape':<30} {'Config':<10} {'BQ':>4} {'BK':>4} {'nw':>3} {'ns':>3} {'ms':>8} {'std':>7}")
print("-" * 80)

for shape_name, sq, sk, nh, hd, causal in attn_shapes:
    results = []
    for BQ, BK, nw, ns, cname in attn_configs:
        try:
            ms, std = run_attn_config(sq, sk, nh, hd, BQ, BK, nw, ns, causal)
            results.append((ms, std, cname, BQ, BK, nw, ns))
        except Exception as e:
            results.append((999, 0, cname, BQ, BK, nw, ns))
    results.sort(key=lambda x: x[0])
    for i, (ms, std, cname, BQ, BK, nw, ns) in enumerate(results):
        winner = " ← WINNER" if i == 0 else ""
        print(f"  {shape_name:<28} {cname:<10} {BQ:>4} {BK:>4} {nw:>3} {ns:>3} "
              f"{ms:>7.3f}ms {std:>6.3f}{winner}")
    print()

print("Done. Paste this output and the Section 5.1 analysis can be written.")

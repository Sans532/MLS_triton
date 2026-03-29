"""
Run this from hw1-asr/ on the cluster AFTER running benchmark.sh once
(so autotune cache is populated).

Usage:
    python print_winning_configs.py glm_asr_triton_template
"""
import sys, os, importlib

folder = sys.argv[1] if len(sys.argv) > 1 else "glm_asr_triton_template"
script_dir = os.path.dirname(os.path.abspath(__file__))
folder_path = os.path.join(script_dir, folder)

sys.path.insert(0, folder_path)
for mod in list(sys.modules):
    if mod in ["layers","attention","rope","model","weight_loader"]:
        del sys.modules[mod]

import torch
import triton

# Import the kernels
import layers
import attention as attn_mod

print("=" * 60)
print("AUTOTUNE WINNING CONFIGS")
print("=" * 60)
print()
print("NOTE: Run benchmark.sh first to populate the autotune cache.")
print("If cache is empty, values will show None.\n")

# linear_kernel_tf32
print("--- linear_kernel_tf32 ---")
try:
    kernel = layers.linear_kernel_tf32
    if hasattr(kernel, 'cache') and kernel.cache:
        for key, config in sorted(kernel.cache.items()):
            M, N, K = key
            print(f"  M={M:5d}, N={N:6d}, K={K:6d}  →  "
                  f"BLOCK_M={config.kwargs['BLOCK_M']:3d}, "
                  f"BLOCK_N={config.kwargs['BLOCK_N']:3d}, "
                  f"BLOCK_K={config.kwargs['BLOCK_K']:3d}, "
                  f"warps={config.num_warps}, "
                  f"stages={config.num_stages}")
    else:
        print("  Cache empty — run benchmark.sh first")
except Exception as e:
    print(f"  Error: {e}")

print()
print("--- attention_fused_kernel ---")
try:
    kernel = attn_mod.attention_fused_kernel
    if hasattr(kernel, 'cache') and kernel.cache:
        for key, config in sorted(kernel.cache.items()):
            print(f"  key={key}  →  "
                  f"BLOCK_Q={config.kwargs['BLOCK_Q']:3d}, "
                  f"BLOCK_K={config.kwargs['BLOCK_K']:3d}, "
                  f"warps={config.num_warps}, "
                  f"stages={config.num_stages}")
    else:
        print("  Cache empty — run benchmark.sh first")
except Exception as e:
    print(f"  Error: {e}")

print()
print("--- swiglu_fused_kernel ---")
try:
    kernel = layers.swiglu_fused_kernel
    if hasattr(kernel, 'cache') and kernel.cache:
        for key, config in sorted(kernel.cache.items()):
            M, N, K = key
            print(f"  M={M:5d}, N={N:6d}, K={K:6d}  →  "
                  f"BLOCK_M={config.kwargs['BLOCK_M']:3d}, "
                  f"BLOCK_N={config.kwargs['BLOCK_N']:3d}, "
                  f"BLOCK_K={config.kwargs['BLOCK_K']:3d}, "
                  f"warps={config.num_warps}, "
                  f"stages={config.num_stages}")
    else:
        print("  Cache empty — run benchmark.sh first")
except Exception as e:
    print(f"  Error: {e}")

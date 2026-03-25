# HW1-ASR: Team Task Tracker

**Track:** Triton  
**Files we own:** `hw1-asr/glm_asr_triton_template/layers.py`, `attention.py`, `rope.py`  
**Do NOT touch:** `model.py`, `conv.py`, `weight_loader.py`  
**Test command:** `./benchmark.sh glm_asr_triton_template` → must show `Accuracy: 100.0% | Status: PASS`

---

## Grading Requirements (from README)

The submission **must** include at least these 3 optimizations — they are checked during grading:

1. **Tile/block size tuning** — try at least 2–3 configurations of `BLOCK_M`, `BLOCK_N`, `BLOCK_K`, `num_warps`, `num_stages` and pick the best for the target GPU
2. **At least 1 fused kernel** — fuse two or more ops that are currently separate to reduce memory roundtrips and kernel launch overhead
3. **FlashAttention-style attention** — streaming softmax, blockwise QK^T, numerically stable, then multiply by V (refactor `attention.py`)

Everything else is bonus performance on top of these three.

---

## Status

### ✅ Done

| What | File | Notes |
|------|------|-------|
| `rmsnorm_kernel` | `layers.py` | `x / sqrt(mean(x²) + eps) * w` |
| `layernorm_kernel` | `layers.py` | `(x - mean) / sqrt(var + eps) * w + b` |
| `gelu_kernel` | `layers.py` | tanh approximation |
| `silu_kernel` | `layers.py` | `x * sigmoid(x)` |
| `linear_kernel_tf32` | `layers.py` | tiled `A @ B` with `tl.dot` |
| `softmax_kernel` | `layers.py` | numerically stable, used in text decoder |
| `attention_scores_kernel` | `attention.py` | `Q @ Kᵀ * scale` |
| `softmax_inplace_kernel` | `attention.py` | in-place softmax over attention scores |
| `attention_output_kernel` | `attention.py` | `attn_weights @ V` |
| `compute_freqs_kernel` | `rope.py` | `cos/sin(pos * inv_freq)`, duplicated for both halves |
| Fused SwiGLU (`swiglu_fused_kernel`) | `layers.py` | already in template, `MLP.FUSED = True` |
| Fused Linear+GELU (`linear_gelu_kernel`) | `layers.py` | already in template, `EncoderMLP.FUSED = True` |

**Baseline correctness: pending first cluster run**  
Expected: `Accuracy: 100.0% | Status: PASS`

---

### 🔲 Required (for grading — must be done)

#### R1. Tile/block size tuning
- [ ] Benchmark `linear_kernel_tf32` with different tile configs on the target GPU
- [ ] Configs to try: `(64,64,32)`, `(128,64,32)`, `(64,128,32)`, `(128,128,32)` for `(BLOCK_M, BLOCK_N, BLOCK_K)`
- [ ] Try `num_warps` in `{2, 4, 8}` and `num_stages` in `{2, 3, 4}`
- [x] Add `@triton.autotune` decorator to `linear_kernel_tf32` to automate this
- [ ] Document best config found and the speedup in the report

```python
# Add above linear_kernel_tf32:
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 64}, num_warps=4, num_stages=4),
    ],
    key=["M", "N", "K"],
)
```

#### R2. FlashAttention-style kernel (biggest single win — ~3.9x from V1→V2 per lecture slides)
- [x] Fuse `attention_scores_kernel` + `softmax_inplace_kernel` + `attention_output_kernel` into one kernel in `attention.py`
- [x] Use online softmax (FA2 algorithm) — eliminates O(n²) HBM materialisation of the full score matrix
- [x] Handle both encoder (bidirectional, no causal mask) and decoder (causal mask) paths
- [x] Keep the existing 3-kernel path as fallback for correctness validation

**Online softmax recurrence (the core of FA2):**
```
# For each K/V block:
m_new = max(m_prev, rowmax(scores_block))
p    = exp(scores_block - m_new)
l_new = exp(m_prev - m_new) * l_prev + rowsum(p)
acc   = acc * exp(m_prev - m_new)[:, None] + tl.dot(p, v_block)
# Final: output = acc / l_new
```

#### R3. At least 1 additional fused kernel (beyond the already-provided ones)
The fused SwiGLU and linear+GELU kernels are already provided in the template — they do not count as "your" fusion. You need at least one more. Easiest options:

- [x] **Option A — Fused RMSNorm + Linear** (decoder path): compute norm and the following linear projection in a single kernel, saving one full read of the hidden state from HBM
- [ ] **Option B — Fused LayerNorm + Linear** (encoder path): same idea for the encoder
- [ ] **Option C — Fused QKV projection**: combine the 3 separate Q, K, V linear projections into one kernel

---

### 🔲 Optional / Bonus (performance improvements for report)

These are not required for correctness or the minimum grade, but will improve your benchmark numbers and give you more to write about in the report. Listed in order of impact (per lecture slide V1→V10 analysis):

| Priority | Optimization | Expected gain | Notes |
|----------|-------------|---------------|-------|
| ⭐⭐⭐ | FlashAttention (R2 above) | ~3.9× | Biggest single win |
| ⭐⭐⭐ | TF32 Tensor Cores in `tl.dot` | ~1.2× | Ensure `allow_tf32=True` (default in recent Triton) |
| ⭐⭐ | Autotune linear (R1 above) | ~1.2× | Free with `@triton.autotune` |
| ⭐⭐ | Autotune fused attention | ~1.1× | Add autotune to FA kernel too |
| ⭐ | Fused norm+linear (R3 above) | ~1.05× | Saves one HBM read per layer |
| ⭐ | Pre-allocated output buffers | ~1.02× | Avoid `torch.empty` inside hot loops |
| ⭐ | FA2 handles GQA natively | ~1.04× | Remove `expand_kv` call in `MultiHeadAttention` |

---

## Implementation Notes

### Architecture reminder
| Component | Layers | Hidden | Heads | Norm | Activation |
|-----------|--------|--------|-------|------|-----------|
| Audio Encoder | 32 | 1280 | 20 (head_dim=64) | LayerNorm | GELU |
| Projector | 2 | 1280→4096→3584 | — | — | GELU |
| Text Decoder | 28 | 3584 | 28Q/4KV GQA (head_dim=128) | RMSNorm | SiLU/SwiGLU |

### Which kernel runs where
- `layernorm` + `gelu` + `linear` → Audio Encoder + Projector  
- `rmsnorm` + `silu` + `linear` → Text Decoder  
- `attention_*` + `compute_freqs` → both Encoder and Decoder  
- Encoder attention: **bidirectional** (no causal mask)  
- Decoder attention: **causal** (upper-triangle masked)  
- Decoder uses **GQA** (28 Q heads, 4 KV heads — `expand_kv` handles this currently)

### Cluster commands
```bash
# Request GPU (H200 node)
srun -p Teaching -w saxa --gres gpu:1 --mem=16G --pty bash

# Activate environment (inside compute node)
source ~/REPO_NAME/utils/setup-triton.sh
cd ~/REPO_NAME/hw1-asr

# Unit tests
cd glm_asr_triton_template && python layers.py && python attention.py && python rope.py

# Correctness + timing
cd .. && ./benchmark.sh glm_asr_triton_template

# Detailed per-operator profiling
./benchmark_detailed.sh glm_asr_triton_template

# Compare against reference example
./benchmark_detailed.sh glm_asr_triton_example

# Nsight profile
./benchmark_detailed.sh glm_asr_triton_template --nsys
```

### Git workflow
```bash
# Always pull before starting
git pull origin main

# Create a branch for your optimization
git checkout -b flash-attention

# Only stage the 3 files we own
git add hw1-asr/glm_asr_triton_template/layers.py \
        hw1-asr/glm_asr_triton_template/attention.py \
        hw1-asr/glm_asr_triton_template/rope.py

git commit -m "feat: fused flashattention kernel"
git push origin flash-attention
# Then open a PR into main
```

---

## Report Checklist (mini-conference format)

- [ ] Abstract: what was implemented, final speedup vs baseline
- [ ] Background: GPU memory hierarchy, Triton programming model, FlashAttention algorithm
- [ ] Implementation: describe each phase with equations (write out online softmax recurrence)
- [ ] Performance table: tokens/sec and latency (ms) for each optimization step
- [ ] Profiling: roofline analysis, HBM bandwidth utilisation from Nsight
- [ ] Conclusion: summary + what FA3/FA4 techniques would need for H100/Blackwell

**Performance table to fill in as you go:**

| Version | Optimization | Avg latency (ms) | Speedup vs V1 |
|---------|-------------|-----------------|---------------|
| V1 | Baseline (this PR) | TBD | 1.0× |
| V2 | + Tile autotune | TBD | TBD |
| V3 | + FlashAttention | TBD | TBD |
| V4 | + Fused norm+linear | TBD | TBD |
| V5 | + Any further opts | TBD | TBD |


---

## Benchmark Results

### Environment
- GPU: NVIDIA H200 (saxa node, Teaching cluster)
- Model: zai-org/GLM-ASR-Nano-2512 (1.5B params)
- Audio: test_audio.wav (3.50s, "Concord returned to its place amidst the tents.")
- Warmup: 1 run, Benchmark: 3 runs

### Results

| Version | Description | Avg Latency | Speed | Accuracy | Status |
|---------|-------------|-------------|-------|----------|--------|
| Reference example | `glm_asr_triton_example` (course baseline) | 1478.3ms ± 0.3ms | 113.71ms/token | 100.0% | PASS |
| V1 — our baseline | `glm_asr_triton_template` (all 10 kernels) | 1437.9ms ± 0.8ms | 110.61ms/token | 100.0% | PASS |

**V1 is already ~3% faster than the reference example.**

### To be filled as optimizations are added

| Version | Optimization | Avg Latency | Speedup vs V1 | Status |
|---------|-------------|-------------|---------------|--------|
| V1 | Baseline (all 10 kernels) | 1437.9ms | 1.00× | ✅ Done |
| V2 | + FlashAttention FA2 | TBD | TBD | ✅ Done |
| V3 | + Autotune linear kernel | TBD | TBD | ✅ Done |
| V4 | + Fused norm+linear | TBD | TBD | ✅ Done |
| V5 | + Any further opts | TBD | TBD | 🔲 |
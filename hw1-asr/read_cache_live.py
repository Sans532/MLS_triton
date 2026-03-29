#!/usr/bin/env python3
"""
read_cache_live.py — runs one inference pass then reads the autotune cache
from the kernel objects while they are still in memory.

Run from hw1-asr/:
    python read_cache_live.py glm_asr_triton_template
"""
import sys, os, time
import torch

folder = sys.argv[1] if len(sys.argv) > 1 else "glm_asr_triton_template"
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(script_dir, folder))

for mod in list(sys.modules):
    if mod in ["layers","attention","rope","model","weight_loader"]:
        del sys.modules[mod]

import triton
import layers
import attention as attn_mod

# ── Run one warmup pass to populate autotune cache ───────────────────────────
print("Loading model and running warmup to populate autotune cache...")

import importlib
from weight_loader import load_model_from_hf

model, processor = load_model_from_hf("zai-org/GLM-ASR-Nano-2512")
device = torch.device("cuda")

import numpy as np, wave, struct

# Load test audio
wav_path = os.path.join(script_dir, "test_audio.wav")
with wave.open(wav_path, "rb") as wf:
    sr = wf.getframerate()
    n  = wf.getnframes()
    ch = wf.getnchannels()
    sw = wf.getsampwidth()
    raw = wf.readframes(n)
audio = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0

inputs = processor.apply_transcription_request(audio, sampling_rate=16000)
feats  = inputs.input_features.to(device, dtype=torch.float32)
ids    = inputs.input_ids.to(device, dtype=torch.int64)
mask   = None
if hasattr(inputs, "input_features_mask") and inputs.input_features_mask is not None:
    mask = inputs.input_features_mask.to(device, dtype=torch.float32)

print("Running warmup inference (populates autotune cache)...")
with torch.no_grad():
    _ = model.generate(feats, input_ids=ids, input_features_mask=mask,
                        max_new_tokens=100, temperature=1.0, top_k=1)
torch.cuda.synchronize()
print("Warmup done.\n")

# ── Now read the autotune cache from kernel objects ───────────────────────────
print("=" * 65)
print("AUTOTUNE WINNING CONFIGS (from live cache)")
print("=" * 65)
print()

def print_cache(kernel, name):
    print(f"--- {name} ---")
    try:
        cache = kernel.cache
        if not cache:
            print("  (empty)")
            return
        for key, cfg in sorted(cache.items()):
            kwargs = cfg.kwargs
            nw     = cfg.num_warps
            ns     = cfg.num_stages
            print(f"  key={key}")
            kw_str = ", ".join(f"{k}={v}" for k, v in sorted(kwargs.items()))
            print(f"    Config: {kw_str}, num_warps={nw}, num_stages={ns}")
    except Exception as e:
        print(f"  Error reading cache: {e}")
    print()

print_cache(layers.linear_kernel_tf32,   "linear_kernel_tf32")
print_cache(layers.swiglu_fused_kernel,  "swiglu_fused_kernel")
print_cache(attn_mod.attention_fused_kernel, "attention_fused_kernel")

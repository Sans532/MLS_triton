#!/usr/bin/env python3
"""
benchmark_multi.py  —  Multi-sample GLM-ASR benchmark
Matches benchmark.sh output: per-run ms, tokens, ms/token, WER, RTF.

Drop into hw1-asr/ (same folder as benchmark.sh).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
QUICK START
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  python benchmark_multi.py glm_asr_triton_template                # 5 samples
  python benchmark_multi.py glm_asr_triton_template --n-samples 10
  python benchmark_multi.py glm_asr_triton_template --n-samples 20
  python benchmark_multi.py glm_asr_triton_template \\
      --compare glm_asr_triton_example --n-samples 10

The script tries three sources in order:
  1. Auto-download from OpenSLR (works on regular internet, blocked on cluster)
  2. HuggingFace datasets API  (pip install datasets, works with HF_TOKEN)
  3. Local folder  hw1-asr/test_samples/audio/  (always works)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SETTING UP LOCAL FILES (cluster / offline use)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Folder layout expected inside hw1-asr/:

    hw1-asr/
    ├── benchmark_multi.py        ← this script
    └── test_samples/
        ├── audio/                ← put .flac OR .wav files here
        │   ├── 61-70968-0000.flac
        │   ├── 1272-128104-0000.flac
        │   └── ...
        └── transcripts.txt       ← one transcript per line, alphabetical

--- Option A: wget individual .flac files (run on your laptop) ----------

  mkdir -p ~/librispeech_flac && cd ~/librispeech_flac
  B=https://www.openslr.org/resources/12/test-clean
  wget -nc $B/61/70968/61-70968-0000.flac
  wget -nc $B/61/70968/61-70968-0001.flac
  wget -nc $B/61/70968/61-70968-0002.flac
  wget -nc $B/1272/128104/1272-128104-0000.flac
  wget -nc $B/1272/128104/1272-128104-0001.flac
  wget -nc $B/1272/128104/1272-128104-0002.flac
  wget -nc $B/1462/170138/1462-170138-0000.flac
  wget -nc $B/1462/170138/1462-170138-0001.flac
  wget -nc $B/1673/143397/1673-143397-0000.flac
  wget -nc $B/1919/142785/1919-142785-0000.flac
  wget -nc $B/2094/142345/2094-142345-0000.flac
  wget -nc $B/2300/131720/2300-131720-0000.flac
  wget -nc $B/2961/960/2961-960-0000.flac
  wget -nc $B/3570/5694/3570-5694-0000.flac
  wget -nc $B/4077/13751/4077-13751-0000.flac
  wget -nc $B/4446/2275/4446-2275-0000.flac
  wget -nc $B/5142/33396/5142-33396-0000.flac
  wget -nc $B/6829/68771/6829-68771-0000.flac
  wget -nc $B/7021/79730/7021-79730-0000.flac
  wget -nc $B/8555/292519/8555-292519-0000.flac

  Then scp to the cluster and place in hw1-asr/test_samples/audio/
  (The script reads .flac directly via ffmpeg which is on the cluster)

--- Option B: HuggingFace (on laptop, needs: pip install datasets) ------

  python3 -c "
  from datasets import load_dataset; import soundfile as sf, os
  ds = load_dataset('librispeech_asr', 'clean', split='test', trust_remote_code=True)
  os.makedirs('librispeech_flac', exist_ok=True)
  for i, item in enumerate(ds):
      if i >= 20: break
      audio = item['audio']
      fname = f'librispeech_flac/{item[\"id\"]}.flac'
      sf.write(fname, audio['array'], audio['sampling_rate'])
      print(fname, '|', item['text'])
  "
  # Then scp to hw1-asr/test_samples/audio/

--- If you have MP3 files — free online converters (no install needed) --

  All three convert to 16 kHz mono 16-bit PCM WAV (required spec):
  • https://cloudconvert.com/mp3-to-wav
    Settings: Sample rate = 16000, Audio channels = 1, Bit depth = 16
  • https://convertio.co/mp3-wav
    Click "Settings" → Frequency: 16000 Hz, Channels: Mono
  • https://audio.online-convert.com/convert-to-wav
    Sampling rate: 16000, Audio channels: 1 (Mono), Codec: PCM 16-bit

  After converting, rename so there are no spaces in filenames.
  Place .wav files in hw1-asr/test_samples/audio/

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TRANSCRIPTS FILE  (hw1-asr/test_samples/transcripts.txt)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
One line per audio file, in alphabetical filename order.
If filenames match the 20 built-in samples the script fills them
automatically — you only need this file for custom audio.

Built-in transcripts (copy into transcripts.txt if needed):
  CONCORD RETURNED TO ITS PLACE AMIDST THE TENTS
  THE COMPANY HAD THE BEST WISHES OF ALL WHO KNEW THEM
  MUCH DEPENDS ON TRAINING
  MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL
  NOR IS MISTER QUILTERS MANNER LESS INTERESTING THAN HIS MATTER
  HE TELLS US THAT AT THIS FESTIVE SEASON OF THE YEAR WITH CHRISTMAS AND ROAST BEEF LOOMING BEFORE US SIMILES DRAWN FROM EATING AND ITS RESULTS OCCUR MOST READILY TO THE MIND
  I HAD ALWAYS LONGED FOR A HORSE AND A GUN AND I COULD NOW HAVE BOTH
  EVERY NIGHT I PREPARED MY PLANS FOR THE MORROW
  IT IS A TRUTH UNIVERSALLY ACKNOWLEDGED THAT A SINGLE MAN IN POSSESSION OF A GOOD FORTUNE MUST BE IN WANT OF A WIFE
  THE YEAR WAS DRAWING TO A CLOSE AND WINTER REIGNED SUPREME OVER THE RUGGED LANDSCAPE
  SHE HAD COME TO LOVE THE DEEP SILENCES OF THE FOREST
  THE OLD GENTLEMAN RAISED HIS EYES AND LOOKED STEADILY AT THE SPEAKER
  THE ONLY THING THAT SEEMED ALIVE WAS THE VOICE OF THE CREEK
  BEFORE ME LAY A SCENE OF INDESCRIBABLE DESOLATION
  HE WAS A TALL SPARE MAN WITH A LONG THIN FACE AND A SHARP NOSE
  IT IS BETTER TO KNOW SOME OF THE QUESTIONS THAN ALL OF THE ANSWERS
  THE MORNING SUN HAD BARELY TOUCHED THE HILLTOPS WHEN HE SET OUT
  NOTHING IN THE WORLD IS MORE DANGEROUS THAN SINCERE IGNORANCE AND CONSCIENTIOUS STUPIDITY
  WHAT A NICE THING TO HAVE SOMEBODY TO EXPLAIN IT ALL
  THE SENTENCE OF DEATH WAS COMMUTED TO ONE OF BANISHMENT
"""

import os, sys, re, time, wave, argparse, importlib, subprocess
# soundfile handles .flac natively — install once: pip install soundfile --user
import numpy as np
from typing import List, Tuple, Optional


# ─── 20 built-in LibriSpeech test-clean samples ─────────────────────────────
# 11 different speakers for diversity in gender, speaking rate, vocabulary
# Format: (url_suffix_on_openslr, expected_transcript, speaker_gender)
BUILTIN_SAMPLES = [
    ("61/70968/61-70968-0000.flac",
     "CONCORD RETURNED TO ITS PLACE AMIDST THE TENTS", "F"),
    ("61/70968/61-70968-0001.flac",
     "THE COMPANY HAD THE BEST WISHES OF ALL WHO KNEW THEM", "F"),
    ("61/70968/61-70968-0002.flac",
     "MUCH DEPENDS ON TRAINING", "F"),
    ("1272/128104/1272-128104-0000.flac",
     "MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL", "M"),
    ("1272/128104/1272-128104-0001.flac",
     "NOR IS MISTER QUILTERS MANNER LESS INTERESTING THAN HIS MATTER", "M"),
    ("1272/128104/1272-128104-0002.flac",
     "HE TELLS US THAT AT THIS FESTIVE SEASON OF THE YEAR WITH CHRISTMAS AND ROAST BEEF LOOMING BEFORE US SIMILES DRAWN FROM EATING AND ITS RESULTS OCCUR MOST READILY TO THE MIND", "M"),
    ("1462/170138/1462-170138-0000.flac",
     "I HAD ALWAYS LONGED FOR A HORSE AND A GUN AND I COULD NOW HAVE BOTH", "M"),
    ("1462/170138/1462-170138-0001.flac",
     "EVERY NIGHT I PREPARED MY PLANS FOR THE MORROW", "M"),
    ("1673/143397/1673-143397-0000.flac",
     "IT IS A TRUTH UNIVERSALLY ACKNOWLEDGED THAT A SINGLE MAN IN POSSESSION OF A GOOD FORTUNE MUST BE IN WANT OF A WIFE", "F"),
    ("1919/142785/1919-142785-0000.flac",
     "THE YEAR WAS DRAWING TO A CLOSE AND WINTER REIGNED SUPREME OVER THE RUGGED LANDSCAPE", "M"),
    ("2094/142345/2094-142345-0000.flac",
     "SHE HAD COME TO LOVE THE DEEP SILENCES OF THE FOREST", "F"),
    ("2300/131720/2300-131720-0000.flac",
     "THE OLD GENTLEMAN RAISED HIS EYES AND LOOKED STEADILY AT THE SPEAKER", "M"),
    ("2961/960/2961-960-0000.flac",
     "THE ONLY THING THAT SEEMED ALIVE WAS THE VOICE OF THE CREEK", "M"),
    ("3570/5694/3570-5694-0000.flac",
     "BEFORE ME LAY A SCENE OF INDESCRIBABLE DESOLATION", "F"),
    ("4077/13751/4077-13751-0000.flac",
     "HE WAS A TALL SPARE MAN WITH A LONG THIN FACE AND A SHARP NOSE", "M"),
    ("4446/2275/4446-2275-0000.flac",
     "IT IS BETTER TO KNOW SOME OF THE QUESTIONS THAN ALL OF THE ANSWERS", "F"),
    ("5142/33396/5142-33396-0000.flac",
     "THE MORNING SUN HAD BARELY TOUCHED THE HILLTOPS WHEN HE SET OUT", "M"),
    ("6829/68771/6829-68771-0000.flac",
     "NOTHING IN THE WORLD IS MORE DANGEROUS THAN SINCERE IGNORANCE AND CONSCIENTIOUS STUPIDITY", "M"),
    ("7021/79730/7021-79730-0000.flac",
     "WHAT A NICE THING TO HAVE SOMEBODY TO EXPLAIN IT ALL", "F"),
    ("8555/292519/8555-292519-0000.flac",
     "THE SENTENCE OF DEATH WAS COMMUTED TO ONE OF BANISHMENT", "F"),
]

# Lookup: filename stem -> (transcript, gender)
BUILTIN_MAP = {
    os.path.splitext(os.path.basename(s))[0]: (t, g)
    for s, t, g in BUILTIN_SAMPLES
}

OPENSLR_BASE = "https://www.openslr.org/resources/12/test-clean"
HF_REPO      = "hf-internal-testing/librispeech_asr_demo"
CACHE_DIR    = os.path.expanduser("~/.cache/glm_asr_samples")


# ─── Audio utilities ─────────────────────────────────────────────────────────

def read_wav(path):
    try:
        with wave.open(path, "rb") as wf:
            sr, n, ch, sw = (wf.getframerate(), wf.getnframes(),
                             wf.getnchannels(), wf.getsampwidth())
            raw = wf.readframes(n)
        if   sw == 2: a = np.frombuffer(raw, np.int16).astype(np.float32) / 32768.0
        elif sw == 4: a = np.frombuffer(raw, np.int32).astype(np.float32) / 2147483648.0
        else: return None, sr
        if ch > 1: a = a.reshape(-1, ch).mean(1)
        return a.astype(np.float32), sr
    except Exception:
        return None, 16000


def resample_audio(audio, sr_in, sr_out=16000):
    if sr_in == sr_out: return audio
    try:
        from scipy import signal as sp
        return sp.resample(audio, int(len(audio)*sr_out/sr_in)).astype(np.float32)
    except ImportError:
        old = np.arange(len(audio))
        new = np.linspace(0, len(audio)-1, int(len(audio)*sr_out/sr_in))
        return np.interp(new, old, audio).astype(np.float32)


def load_audio_file(path):
    """
    Load any audio file as 16 kHz mono float32.
    Priority:
      1. soundfile  — handles .flac, .wav, .mp3, .ogg natively (pure Python, no system tool)
      2. stdlib wave — .wav only fallback if soundfile not installed
    No ffmpeg required.
    Install soundfile once on the cluster: pip install soundfile --user
    """
    try:
        import soundfile as sf
        a, sr = sf.read(path, dtype="float32", always_2d=False)
        if a.ndim > 1:
            a = a.mean(axis=1)   # stereo -> mono
        return resample_audio(a.astype(np.float32), sr)
    except ImportError:
        pass   # soundfile not installed — try stdlib wave for .wav
    except Exception as e:
        print(f"  soundfile error on {os.path.basename(path)}: {e}")
        return None

    # stdlib wave fallback (.wav only)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".wav":
        a, sr = read_wav(path)
        if a is not None:
            return resample_audio(a, sr)

    print(f"  Cannot read {os.path.basename(path)}: "
          "install soundfile first:  pip install soundfile --user")
    return None


# ─── Download helpers ─────────────────────────────────────────────────────────

def try_openslr(suffix):
    """Try downloading one .flac from OpenSLR. Returns local path or None."""
    import urllib.request
    os.makedirs(CACHE_DIR, exist_ok=True)
    local = os.path.join(CACHE_DIR, suffix.replace("/", "_"))
    if os.path.exists(local):
        return local
    try:
        urllib.request.urlretrieve(f"{OPENSLR_BASE}/{suffix}", local)
        return local
    except Exception:
        try: os.remove(local)
        except: pass
        return None


def try_hf_datasets(n):
    """
    Try downloading via HuggingFace datasets library.
    Returns list of (audio_array, label, transcript) or empty list.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        return []

    try:
        ds = load_dataset("librispeech_asr", "clean", split="test",
                          trust_remote_code=True, streaming=True)
        results = []
        for i, item in enumerate(ds):
            if i >= n: break
            try:
                audio = item["audio"]
                arr   = np.array(audio["array"], dtype=np.float32)
                sr    = audio["sampling_rate"]
                arr   = resample_audio(arr, sr)
                label = item.get("id", f"sample_{i:03d}")
                text  = item.get("text", "").upper().strip()
                results.append((arr, label, text))
                print(f"    HF [{i+1}/{n}] {label}  ({len(arr)/16000:.1f}s)")
            except Exception:
                continue
        return results
    except Exception:
        return []


# ─── Sample loading ───────────────────────────────────────────────────────────

def load_samples(n, script_dir, audio_dir=None, transcripts_file=None):
    """
    Load n samples using three-tier fallback:
      Tier 1: OpenSLR auto-download
      Tier 2: HuggingFace datasets API
      Tier 3: Local hw1-asr/test_samples/audio/ folder
    Returns list of (audio_array, expected_text, label).
    """
    samples = []

    # ── Tier 3 candidate: local folder ───────────────────────────────────────
    if audio_dir is None:
        default_dir = os.path.join(script_dir, "test_samples", "audio")
        if os.path.isdir(default_dir):
            audio_dir = default_dir

    if transcripts_file is None:
        default_tx = os.path.join(script_dir, "test_samples", "transcripts.txt")
        if os.path.exists(default_tx):
            transcripts_file = default_tx

    # If --audio-dir was specified, use it directly (skip download)
    if audio_dir and os.path.isdir(audio_dir):
        # Auto-discover transcripts.txt next to audio dir if not explicitly given
        if transcripts_file is None:
            sibling_tx = os.path.join(os.path.dirname(audio_dir.rstrip("/")), "transcripts.txt")
            if os.path.exists(sibling_tx):
                transcripts_file = sibling_tx
                print(f"  Auto-found transcripts: {sibling_tx}")

        print(f"\n[Tier 3] Loading from {audio_dir}")
        files = sorted(
            os.path.join(audio_dir, f) for f in os.listdir(audio_dir)
            if f.lower().endswith((".wav", ".flac", ".mp3"))
        )[:n]

        # Load transcripts as {id: text} dict — supports both formats:
        #   "ID TRANSCRIPT"  (new format, order-independent)
        #   "TRANSCRIPT"     (old format, alphabetical order)
        tx_map   = {}
        tx_lines = []
        if transcripts_file and os.path.exists(transcripts_file):
            with open(transcripts_file) as tf:
                for line in tf:
                    line = line.strip()
                    if not line: continue
                    parts = line.split(" ", 1)
                    # Detect "ID TEXT" format: first token looks like an utterance id
                    # (e.g. "2961-960-0000") — contains hyphens and no spaces before first space
                    if len(parts) == 2 and "-" in parts[0] and parts[0].replace("-","").isalnum():
                        tx_map[parts[0]] = parts[1]
                    else:
                        tx_lines.append(line)   # old line-ordered format

        for i, fpath in enumerate(files):
            audio = load_audio_file(fpath)
            if audio is None:
                print(f"  WARN: cannot read {os.path.basename(fpath)}")
                continue
            stem     = os.path.splitext(os.path.basename(fpath))[0]
            # Transcript priority: id-keyed map > ordered line > built-in map > empty
            if stem in tx_map:
                expected = tx_map[stem]
            elif i < len(tx_lines) and tx_lines[i]:
                expected = tx_lines[i]
            elif stem in BUILTIN_MAP:
                expected = BUILTIN_MAP[stem][0]
            else:
                expected = ""
            print(f"  OK  {stem}  ({len(audio)/16000:.1f}s)  "
                  + (f"transcript: {expected[:55]}.." if len(expected) > 55
                     else f"transcript: {expected or '(none)'}"))
            samples.append((audio, expected, stem))

        if samples:
            return samples
        print("  (no readable audio files found in that directory)")

    # ── Tier 1: OpenSLR auto-download ─────────────────────────────────────────
    print(f"\n[Tier 1] Trying OpenSLR auto-download ({n} files)...")
    for suffix, expected, _ in BUILTIN_SAMPLES[:n]:
        label = os.path.splitext(os.path.basename(suffix))[0]
        print(f"  Downloading {label}...", end="", flush=True)
        local = try_openslr(suffix)
        if local is None:
            print(" blocked")
            break  # cluster proxy blocks all → skip to Tier 2
        audio = load_audio_file(local)
        if audio is None:
            print(" load error")
            continue
        print(f" OK ({len(audio)/16000:.1f}s)")
        samples.append((audio, expected, label))

    if len(samples) == n:
        return samples

    # ── Tier 2: HuggingFace datasets API ──────────────────────────────────────
    remaining = n - len(samples)
    if remaining > 0:
        print(f"\n[Tier 2] Trying HuggingFace datasets API ({remaining} remaining)...")
        hf = try_hf_datasets(remaining)
        if hf:
            # Match HF items to built-in transcripts where possible
            for arr, label, text in hf:
                if not text and label in BUILTIN_MAP:
                    text = BUILTIN_MAP[label][0]
                samples.append((arr, text, label))
            print(f"  Got {len(hf)} sample(s) from HuggingFace")
        else:
            print("  HuggingFace unavailable (try: pip install datasets)")

    # ── Final fallback: local test_audio.wav ──────────────────────────────────
    if not samples:
        wav = os.path.join(script_dir, "test_audio.wav")
        if os.path.exists(wav):
            print(f"\n[Fallback] Using {wav} (all download methods failed)")
            print("  To use multiple samples: place .flac/.wav files in")
            print(f"  {os.path.join(script_dir, 'test_samples', 'audio/')}")
            audio = load_audio_file(wav)
            if audio:
                samples = [(audio, "CONCORD RETURNED TO ITS PLACE AMIDST THE TENTS",
                            "test_audio")]

    return samples


# ─── WER / accuracy ───────────────────────────────────────────────────────────

def word_error_rate(hyp, ref):
    def norm(s): return re.sub(r"[^\w\s]", "", s.upper().strip()).split()
    h, r = norm(hyp), norm(ref)
    if not r: return 0.0 if not h else 1.0
    d = np.zeros((len(r)+1, len(h)+1), dtype=int)
    d[:,0] = np.arange(len(r)+1)
    d[0,:] = np.arange(len(h)+1)
    for i in range(1, len(r)+1):
        for j in range(1, len(h)+1):
            c = 0 if r[i-1] == h[j-1] else 1
            d[i,j] = min(d[i-1,j]+1, d[i,j-1]+1, d[i-1,j-1]+c)
    return d[len(r), len(h)] / len(r)


def word_accuracy(hyp, ref):
    """Word-overlap accuracy — matches benchmark.sh check_transcription."""
    def norm(s): return set(re.sub(r"[^\w\s]", "", s.upper().strip()).split())
    h, r = norm(hyp), norm(ref)
    if not r: return 1.0
    return len(h & r) / len(r)


# ─── Model loading ────────────────────────────────────────────────────────────

def load_model(folder_name, script_dir):
    folder_path = os.path.join(script_dir, folder_name)
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    if folder_path in sys.path: sys.path.remove(folder_path)
    sys.path.insert(0, folder_path)

    for mod in list(sys.modules):
        if mod in ["weight_loader","model","layers","attention","rope","conv"]:
            del sys.modules[mod]

    if "example" in folder_name.lower():
        layers = importlib.import_module("layers")
        layers.Linear.BACKEND = "cublas"
        layers.MLP.FUSED = False
        if hasattr(layers, "EncoderMLP"): layers.EncoderMLP.FUSED = False

    from weight_loader import load_model_from_hf
    model, processor = load_model_from_hf("zai-org/GLM-ASR-Nano-2512")
    return model, processor, folder_path


# ─── Single-sample timed run (identical to benchmark_student.py) ─────────────

def run_one(model, processor, audio, device, num_warmup, num_runs):
    """Warmup + timed runs. Uses torch.cuda.synchronize() + perf_counter.
    Input preparation mirrors prepare_inputs_torch in benchmark_student.py exactly."""
    import torch

    # ── prepare inputs (same logic as benchmark_student.py) ──────────────────
    if hasattr(processor, "apply_transcription_request"):
        inputs = processor.apply_transcription_request(audio, sampling_rate=16000)
        feats  = inputs.input_features.to(device=device, dtype=torch.float32)
        ids    = inputs.input_ids.to(device=device, dtype=torch.int64)
        mask   = None
        if hasattr(inputs, "input_features_mask") and inputs.input_features_mask is not None:
            mask = inputs.input_features_mask.to(device=device, dtype=torch.float32)
    else:
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        feats  = inputs["input_features"].to(device=device, dtype=torch.float32)
        ids    = inputs.get("input_ids")
        if ids is not None:
            ids = ids.to(device=device, dtype=torch.int64)
        mask   = None   # no mask in this path

    def _gen():
        with torch.no_grad():
            return model.generate(feats, input_ids=ids, input_features_mask=mask,
                                   max_new_tokens=100, temperature=1.0, top_k=1)

    print(f"  Warmup ({num_warmup} run(s))...")
    for _ in range(num_warmup):
        _gen()
        if torch.cuda.is_available(): torch.cuda.synchronize()

    print(f"  Benchmarking ({num_runs} run(s))...")
    run_times, output = [], None
    for i in range(num_runs):
        if torch.cuda.is_available(): torch.cuda.synchronize()
        t0 = time.perf_counter()
        output = _gen()
        if torch.cuda.is_available(): torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1000
        run_times.append(ms)
        toks = output.shape[1] - (ids.shape[1] if ids is not None else 0)
        print(f"    Run {i+1}: {ms:.1f}ms ({toks} tokens)")

    gen_np = output.detach().cpu().numpy()
    try:
        trans = processor.batch_decode(gen_np, skip_special_tokens=True)
        trans = trans[0] if isinstance(trans, list) else trans
        if "Please transcribe" in trans:
            trans = trans.split("Please transcribe this audio into text")[-1].strip()
    except Exception:
        try: trans = processor.tokenizer.decode(gen_np[0], skip_special_tokens=True)
        except: trans = "[decode error]"

    mean_ms = float(np.mean(run_times))
    return {
        "mean_ms":       mean_ms,
        "std_ms":        float(np.std(run_times)),
        "tokens":        toks,
        "ms_per_token":  mean_ms / toks if toks > 0 else 0.0,
        "transcription": trans.strip(),
        "run_times":     run_times,
        "duration_s":    len(audio) / 16000,
    }


# ─── Folder benchmark ─────────────────────────────────────────────────────────

def benchmark_folder(folder_name, script_dir, samples, num_warmup, num_runs):
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    SEP = "=" * 70
    print(f"\n{SEP}\nTesting: {folder_name}\n{SEP}")
    print("Loading model...")
    model, processor, folder_path = load_model(folder_name, script_dir)

    results = []
    for idx, (audio, expected, label) in enumerate(samples):
        dur = len(audio) / 16000
        print(f"\n[{idx+1}/{len(samples)}]  {label}  ({dur:.2f}s)")
        if expected:
            print(f"  Expected: {expected}")

        r = run_one(model, processor, audio, device, num_warmup, num_runs)

        wer = word_error_rate(r["transcription"], expected) if expected else None
        acc = word_accuracy(r["transcription"], expected)   if expected else None
        r.update({"label": label, "expected": expected, "wer": wer, "accuracy": acc})

        rtf = r["mean_ms"] / 1000 / dur
        print(f"\n  {'─'*50}")
        print(f"  Time:          {r['mean_ms']:.1f}ms (+/- {r['std_ms']:.1f}ms)")
        print(f"  Tokens:        {r['tokens']}")
        print(f"  Speed:         {r['ms_per_token']:.2f}ms/token")
        print(f"  RTF:           {rtf:.3f}x  "
              f"({'faster' if rtf < 1 else 'slower'} than real-time)")
        print(f"  Transcription: {r['transcription']}")
        if expected:
            status = "PASS" if (acc is not None and acc >= 0.8) else "FAIL"
            wer_s  = f"{wer*100:.1f}%" if wer is not None else "N/A"
            print(f"  Accuracy:      {acc*100:.1f}%   WER: {wer_s}   [{status}]")

        results.append(r)

    try: sys.path.remove(folder_path)
    except ValueError: pass
    return results


# ─── Summary ──────────────────────────────────────────────────────────────────

def print_summary(folder_name, results):
    if not results: return
    lat    = [r["mean_ms"]       for r in results]
    mpt    = [r["ms_per_token"]  for r in results]
    toks   = [r["tokens"]        for r in results]
    wers   = [r["wer"]  for r in results if r["wer"]  is not None]
    rtfs   = [r["mean_ms"]/1000/r["duration_s"]
              for r in results if r["duration_s"] > 0]
    accs   = [r["accuracy"] for r in results if r["accuracy"] is not None]
    passed = sum(1 for a in accs if a >= 0.8)

    # Speaker diversity info
    speakers = set()
    for _, _, lbl in [(None, None, r["label"]) for r in results]:
        parts = lbl.split("-")
        if len(parts) >= 2: speakers.add(parts[0])

    print(f"\n{'='*70}")
    print(f"SUMMARY — {folder_name}  ({len(results)} samples)")
    print(f"{'='*70}")
    if speakers:
        print(f"  Speakers:             {len(speakers)}  ({', '.join(sorted(speakers))})")
    print(f"  Mean latency:         {np.mean(lat):.1f}ms ± {np.std(lat):.1f}ms")
    print(f"  Min / Max latency:    {np.min(lat):.1f}ms / {np.max(lat):.1f}ms")
    print(f"  Mean tokens/sample:   {np.mean(toks):.1f}")
    print(f"  Mean ms/token:        {np.mean(mpt):.2f}ms")
    if rtfs:
        print(f"  Mean RTF:             {np.mean(rtfs):.3f}x  "
              f"({'faster' if np.mean(rtfs) < 1 else 'slower'} than real-time)")
    if accs:
        print(f"  Passed (acc ≥ 80%):   {passed}/{len(results)}")
    if wers:
        print(f"  Mean WER:             {np.mean(wers)*100:.1f}%")
        print(f"  Mean accuracy:        {(1-np.mean(wers))*100:.1f}%")


def print_comparison(ra, na, rb, nb):
    W = 36
    print(f"\n{'='*70}")
    print(f"COMPARISON   {na}   vs   {nb}")
    print(f"{'='*70}")
    print(f"  {'Sample':<{W}}  {na:>14}   {nb:>14}   {'Speedup':>7}")
    print(f"  {'-'*W}  {'-'*14}   {'-'*14}   {'-'*7}")
    for r1, r2 in zip(ra, rb):
        sp = r2["mean_ms"]/r1["mean_ms"] if r1["mean_ms"] > 0 else float("nan")
        print(f"  {r1['label']:<{W}}  {r1['mean_ms']:>12.1f}ms"
              f"   {r2['mean_ms']:>12.1f}ms   {sp:>6.2f}x")
    ma = np.mean([r["mean_ms"] for r in ra])
    mb = np.mean([r["mean_ms"] for r in rb])
    sp = mb/ma if ma > 0 else float("nan")
    print(f"  {'─'*W}  {'─'*14}   {'─'*14}   {'─'*7}")
    print(f"  {'OVERALL':<{W}}  {ma:>12.1f}ms   {mb:>12.1f}ms   {sp:>6.2f}x")
    wa = [r["wer"] for r in ra if r["wer"] is not None]
    wb = [r["wer"] for r in rb if r["wer"] is not None]
    if wa and wb:
        print(f"\n  WER — {na}: {np.mean(wa)*100:.1f}%   "
              f"{nb}: {np.mean(wb)*100:.1f}%")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Multi-sample GLM-ASR benchmark — matches benchmark.sh output format"
    )
    ap.add_argument("folder",
                    help="e.g. glm_asr_triton_template")
    ap.add_argument("--compare", default=None,
                    help="Second folder, e.g. glm_asr_triton_example")
    ap.add_argument("--n-samples", type=int, default=5,
                    choices=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],
                    help="Number of samples: 5 / 10 / 15 / 20 (default: 5)")
    ap.add_argument("--audio-dir", default=None,
                    help="Override audio source. Path to directory of .wav/.flac/.mp3 files. "
                         "If not set, auto-discovers hw1-asr/test_samples/audio/")
    ap.add_argument("--transcripts", default=None,
                    help="Override transcripts file. One line per file alphabetically. "
                         "If not set, auto-discovers hw1-asr/test_samples/transcripts.txt")
    ap.add_argument("--warmup", type=int, default=1,
                    help="Warmup runs per sample (default 1, same as benchmark.sh)")
    ap.add_argument("--runs",   type=int, default=3,
                    help="Timed runs per sample (default 3, same as benchmark.sh)")
    args = ap.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    print(f"\n{'='*70}")
    print(f"GLM-ASR Multi-Sample Benchmark  |  n={args.n_samples} samples")
    print(f"{'='*70}")

    samples = load_samples(
        n=args.n_samples,
        script_dir=script_dir,
        audio_dir=args.audio_dir,
        transcripts_file=args.transcripts,
    )

    if not samples:
        print("\nERROR: No samples could be loaded.")
        print("Options:")
        print(f"  1. Place audio files in {os.path.join(script_dir, 'test_samples', 'audio/')}")
        print("     (See script header for download instructions)")
        print("  2. Use --audio-dir /path/to/your/wavs")
        print("  3. Ensure internet access for auto-download (blocked on cluster)")
        sys.exit(1)

    # Speaker diversity summary
    speakers = sorted(set(
        l.split("-")[0] for _, _, l in samples if "-" in l
    ))
    female = sum(1 for _, _, l in samples
                 if l in BUILTIN_MAP and BUILTIN_MAP[l][1] == "F")
    male   = len(samples) - female
    print(f"\n{len(samples)} sample(s) loaded — "
          f"{len(speakers)} speaker(s)  "
          + (f"[{female}F / {male}M]" if female + male == len(samples) else ""))

    # Run benchmark(s)
    ra = benchmark_folder(args.folder, script_dir, samples, args.warmup, args.runs)
    print_summary(args.folder, ra)

    if args.compare:
        rb = benchmark_folder(args.compare, script_dir, samples, args.warmup, args.runs)
        print_summary(args.compare, rb)
        print_comparison(ra, args.folder, rb, args.compare)


if __name__ == "__main__":
    main()
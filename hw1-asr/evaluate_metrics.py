#!/usr/bin/env python3
"""
Comprehensive Evaluation Suite for GLM-ASR on the H200 or similar GPUs.
Tracks: WER, CER, RTF, Inter-Token Latency, and Peak VRAM.
"""

import argparse
import time
import sys
import os
import torch
import numpy as np
import evaluate
import wave
import struct
import logging

def prepare_inputs_torch(audio_array, processor, device):
    """Reuse the standard testing tensor preparation exactly."""
    if hasattr(processor, 'apply_transcription_request'):
        inputs = processor.apply_transcription_request(audio_array)
        input_features = inputs.input_features.to(device=device, dtype=torch.float32)
        input_ids = inputs.input_ids.to(device=device, dtype=torch.int64)
        input_features_mask = None
        if hasattr(inputs, 'input_features_mask') and inputs.input_features_mask is not None:
            input_features_mask = inputs.input_features_mask.to(device=device, dtype=torch.float32)
    else:
        features = processor(audio_array, sampling_rate=16000, return_tensors="pt", padding="max_length")
        input_features = features['input_features'].to(device=device, dtype=torch.float32)

        mel_frames = input_features.shape[-1]
        num_audio_tokens = max(1, mel_frames // 2 // 4)

        user_token_id = 59253
        assistant_token_id = 59254
        begin_audio_token_id = 59261
        end_audio_token_id = 59262
        audio_token_id = 59260
        newline_token_id = 10
        prompt_token_ids = [9249, 70891, 419, 7122, 1119, 1467]

        input_ids_list = [user_token_id, newline_token_id, begin_audio_token_id]
        input_ids_list.extend([audio_token_id] * num_audio_tokens)
        input_ids_list.extend([end_audio_token_id, user_token_id, newline_token_id])
        input_ids_list.extend(prompt_token_ids)
        input_ids_list.extend([assistant_token_id, newline_token_id])

        input_ids = torch.tensor([input_ids_list], dtype=torch.int64, device=device)
        input_features_mask = None

    return input_features, input_ids, input_features_mask


def decode_output(generated_np, processor):
    try:
        if hasattr(processor, 'tokenizer'):
            transcription = processor.tokenizer.decode(generated_np[0], skip_special_tokens=True)
        elif hasattr(processor, 'decode'):
            transcription = processor.decode(generated_np[0], skip_special_tokens=True)
        else:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained("zai-org/GLM-ASR-Nano-2512", trust_remote_code=True)
            transcription = tokenizer.decode(generated_np[0], skip_special_tokens=True)

        if "Please transcribe this audio into text" in transcription:
            transcription = transcription.split("Please transcribe this audio into text")[-1].strip()
        return transcription
    except Exception as e:
        return f"[decode error: {e}]"


def read_local_wav(filepath):
    """Built-in library wave reader to avoid FFMPEG/libsnd dependencies on Node"""
    with wave.open(filepath, 'rb') as wav:
        sr = wav.getframerate()
        n_channels = wav.getnchannels()
        n_frames = wav.getnframes()
        sample_width = wav.getsampwidth()
        raw_data = wav.readframes(n_frames)
        
        if sample_width == 2:
            fmt = f'<{n_frames * n_channels}h'
            audio = np.array(struct.unpack(fmt, raw_data), dtype=np.float32)
            audio = audio / 32768.0
        else:
            raise ValueError(f"Unsupported sample width: {sample_width}")
        
        if n_channels > 1:
            audio = audio.reshape(-1, n_channels).mean(axis=1)
        return audio, sr


def main():
    parser = argparse.ArgumentParser(description="Evaluate GLM-ASR implementation on standard dummy dataset.")
    parser.add_argument("folder", type=str, help="Folder containing model implementations (e.g. glm_asr_triton_template)")
    parser.add_argument("--max_samples", type=int, default=None, help="Max test dataset samples to process")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(script_dir, args.folder)
    sys.path.insert(0, folder_path)

    # Load Model directly off disk
    print(f"Loading native models from {args.folder}...")
    from weight_loader import load_model_from_hf
    model, processor = load_model_from_hf("zai-org/GLM-ASR-Nano-2512")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Evaluators
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    dataset_dir = os.path.join(script_dir, "dummy_dataset")
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset folder '{dataset_dir}' not found.")
        print("Please run 'python download_dataset.py' to fetch and format the local dataset.")
        sys.exit(1)

    print(f"\nLoading local dataset from '{dataset_dir}'...")
    with open(os.path.join(dataset_dir, "transcripts.txt"), "r", encoding="utf-8") as f:
        ds_lines = [line.strip() for line in f if line.strip()]
        
    if args.max_samples:
        ds_lines = ds_lines[:args.max_samples]

    generate_fn = model.generate
    if hasattr(model, 'generate_v8b'): generate_fn = model.generate_v8b
    elif hasattr(model, 'generate_v8'): generate_fn = model.generate_v8
    elif hasattr(model, 'generate_v6'): generate_fn = model.generate_v6

    # Tracking lists
    predictions = []
    references = []
    
    total_audio_duration = 0.0
    total_inference_time = 0.0
    total_tokens = 0

    print("Running GPU warmup (ensures Triton kernel JIT compiling doesn't skew first dataset inference)...")
    warmup_filename, _ = ds_lines[0].split("|", 1)
    warmup_audio, _ = read_local_wav(os.path.join(dataset_dir, warmup_filename))
    w_in_feat, w_in_ids, w_in_mask = prepare_inputs_torch(warmup_audio, processor, device)
    
    with torch.no_grad():
        try:
            _ = generate_fn(w_in_feat, input_ids=w_in_ids, input_features_mask=w_in_mask, max_new_tokens=20, temperature=1.0, top_k=1)
        except TypeError:
            _ = generate_fn(w_in_feat, input_ids=w_in_ids, max_new_tokens=20, temperature=1.0, top_k=1)
    
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    
    print("\n" + "="*50)
    print("STARTING BATCH EVALUATION")
    print("="*50)
    
    for i, line in enumerate(ds_lines):
        filename, expected_text = line.split("|", 1)
        audio_path = os.path.join(dataset_dir, filename)
        
        audio_array, sr = read_local_wav(audio_path)
        
        # Double check sanity resample (should be 16k out of box via download script)
        if sr != 16000:
            old_indices = np.arange(len(audio_array))
            new_length = int(len(audio_array) * 16000 / sr)
            new_indices = np.linspace(0, len(audio_array) - 1, new_length)
            audio_array = np.interp(new_indices, old_indices, audio_array).astype(np.float32)
        
        duration = len(audio_array) / 16000
        total_audio_duration += duration
        
        input_features, input_ids, input_features_mask = prepare_inputs_torch(audio_array, processor, device)
        
        torch.cuda.synchronize()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        with torch.no_grad():
            try:
                output = generate_fn(input_features, input_ids=input_ids, input_features_mask=input_features_mask, max_new_tokens=100, temperature=1.0, top_k=1)
            except TypeError:
                output = generate_fn(input_features, input_ids=input_ids, max_new_tokens=100, temperature=1.0, top_k=1)
        end_event.record()
        torch.cuda.synchronize()
        
        inf_time = start_event.elapsed_time(end_event) / 1000.0
        total_inference_time += inf_time
        
        generated_tokens = output.shape[1] - input_ids.shape[1]
        total_tokens += generated_tokens
        
        output_np = output.detach().cpu().numpy()
        transcription = decode_output(output_np, processor)
        
        predictions.append(transcription.upper().strip())
        references.append(expected_text.upper().strip())
        
        rtf = inf_time / duration
        itl = (inf_time * 1000) / generated_tokens if generated_tokens > 0 else 0
        print(f"Sample {i+1:02d} | RTF: {rtf:.3f} | ITL: {itl:.1f}ms/t | Transcribed: {transcription}")

    # Standardize arrays to prevent metric crashing on empty lines
    for i in range(len(predictions)):
        if predictions[i] == "": predictions[i] = "EMPTY"
        if references[i] == "": references[i] = "EMPTY"

    wer = wer_metric.compute(predictions=predictions, references=references)
    cer = cer_metric.compute(predictions=predictions, references=references)
    
    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 3)
    avg_rtf = total_inference_time / total_audio_duration
    avg_itl = (total_inference_time * 1000) / total_tokens if total_tokens > 0 else 0
    
    print("\n" + "="*50)
    print(f"H200 EVALUATION RESULTS - {args.folder}")
    print("="*50)
    print(f"Model Accuracy (WER):     {wer * 100:.2f}%")
    print(f"Model Accuracy (CER):     {cer * 100:.2f}%")
    print(f"Real-Time Factor (RTF):   {avg_rtf:.3f}")
    print(f"Average Gen Latency:      {avg_itl:.2f} ms/token")
    print(f"Peak VRAM Usage:          {peak_vram:.2f} GB")
    print(f"Total Audio Processed:    {total_audio_duration:.2f} s")
    print(f"Total Inference Time:     {total_inference_time:.2f} s")
    print("="*50)


if __name__ == "__main__":
    main()
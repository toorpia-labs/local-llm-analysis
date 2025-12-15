#!/usr/bin/env python3
"""
LLM Hidden State Extraction with STFT-style Overlapping Segmentation

Generates text from an LLM multiple times and extracts hidden states,
normalizing variable-length token sequences into fixed-size segments
using overlapping windows (similar to STFT).

Author: Experiment Framework
Version: 1.0.0
"""

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
import torch
from scipy.signal import get_window
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig
)


def get_device() -> torch.device:
    """Auto-detect available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def create_overlapping_segments(
    hidden_states: np.ndarray,
    n_segments: int,
    overlap: float = 0.5,
    window_func: str = "hann"
) -> np.ndarray:
    """
    Create fixed-size segments from variable-length hidden states using
    STFT-style overlapping windows.

    Args:
        hidden_states: (num_tokens, hidden_dim) array
        n_segments: Fixed number of output segments
        overlap: Overlap ratio (0.0-0.9)
        window_func: Window function type ("rect", "hann", "hamming")

    Returns:
        segments: (n_segments, hidden_dim) array
    """
    num_tokens, hidden_dim = hidden_states.shape

    # Step 1: Calculate segment center positions (normalized 0-1)
    segment_centers = np.linspace(0, 1, n_segments)

    # Step 2: Calculate window width
    if n_segments > 1:
        hop = 1.0 / (n_segments - 1)
        window_width = hop / (1 - overlap)
    else:
        window_width = 1.0

    # Step 3: Calculate each segment
    segments = np.zeros((n_segments, hidden_dim))

    for seg_idx, center in enumerate(segment_centers):
        # Window range (normalized coordinates)
        win_start = max(0, center - window_width / 2)
        win_end = min(1, center + window_width / 2)

        # Convert to token indices
        token_start = int(win_start * (num_tokens - 1))
        token_end = int(np.ceil(win_end * (num_tokens - 1))) + 1
        token_end = min(token_end, num_tokens)

        # Window length in tokens
        win_length = token_end - token_start
        if win_length < 1:
            win_length = 1
            token_end = token_start + 1

        # Generate window function
        if window_func == "rect":
            window = np.ones(win_length)
        else:
            window = get_window(window_func, win_length)

        # Weighted average of tokens in window
        windowed_states = hidden_states[token_start:token_end] * window[:, np.newaxis]
        segments[seg_idx] = windowed_states.sum(axis=0) / window.sum()

    return segments


def compute_segment_metadata(
    num_tokens: int,
    n_segments: int,
    overlap: float
) -> Dict[str, Any]:
    """Calculate segmentation metadata."""
    if n_segments > 1:
        hop = 1.0 / (n_segments - 1)
        window_width = hop / (1 - overlap)
    else:
        hop = 1.0
        window_width = 1.0

    # Effective values in token units
    effective_window_tokens = window_width * (num_tokens - 1)
    effective_hop_tokens = hop * (num_tokens - 1)

    return {
        "original_num_tokens": num_tokens,
        "normalized_hop": float(hop),
        "normalized_window_width": float(window_width),
        "effective_window_tokens": round(float(effective_window_tokens), 2),
        "effective_hop_tokens": round(float(effective_hop_tokens), 2),
        "overlap_ratio": float(overlap)
    }


def extract_hidden_states_from_generation(
    outputs,
    input_length: int,
    layer_idx: int = -1
) -> np.ndarray:
    """
    Extract hidden states for newly generated tokens only.

    Args:
        outputs: Model generation output with hidden_states
        input_length: Length of input prompt tokens
        layer_idx: Layer to extract (-1 for last layer)

    Returns:
        hidden_states: (num_new_tokens, hidden_dim) array
    """
    # outputs.hidden_states is tuple of tuples:
    # hidden_states[token_idx][layer_idx] -> (batch, seq_len, hidden_dim)

    hidden_states_list = []

    for token_idx in range(len(outputs.hidden_states)):
        # Get hidden state for this generation step
        step_hidden = outputs.hidden_states[token_idx]

        # Extract specified layer
        layer_hidden = step_hidden[layer_idx]  # (batch, seq_len, hidden_dim)

        # Get the last position (newly generated token)
        # For first token: seq_len = input_length + 1
        # For nth token: seq_len = input_length + n
        token_hidden = layer_hidden[0, -1, :].cpu().numpy()  # (hidden_dim,)

        hidden_states_list.append(token_hidden)

    # Stack into (num_new_tokens, hidden_dim)
    hidden_states = np.stack(hidden_states_list, axis=0)

    return hidden_states


def run_extraction(args):
    """Main extraction function."""

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("LLM Hidden State Extraction")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Prompt: '{args.prompt}'")
    print(f"Trials: {args.n_trials}")
    print(f"Segments: {args.n_segments} (overlap={args.overlap})")
    print(f"Output: {output_dir}")
    print("="*60)

    # Device setup
    device = get_device()
    print(f"Device: {device}")

    # Load model and tokenizer
    print(f"\nLoading model: {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if args.fp16 else torch.float32,
        trust_remote_code=True,
        device_map="auto" if device.type == "cuda" else None
    )

    if device.type != "cuda":
        model = model.to(device)

    model.eval()

    # Set pad token if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Get model info
    config = model.config
    hidden_dim = getattr(config, 'hidden_size', getattr(config, 'n_embd', None))
    num_layers = getattr(config, 'num_hidden_layers', getattr(config, 'n_layer', None))

    print(f"Hidden dim: {hidden_dim}")
    print(f"Num layers: {num_layers}")
    print(f"Extracting layer: {args.layer}")

    # Tokenize prompt
    inputs = tokenizer(args.prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    input_length = input_ids.shape[1]

    print(f"Input tokens: {input_length}")

    # Generation config
    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        output_hidden_states=True,
        return_dict_in_generate=True
    )

    # Storage
    all_segments = []
    trial_metadata = []
    generations_log = []
    raw_hidden_states = {}

    skipped_trials = 0
    timestamp_start = datetime.now()

    # Run trials
    print(f"\nRunning {args.n_trials} trials...")

    with torch.no_grad():
        for trial_id in tqdm(range(args.n_trials), desc="Generating"):
            trial_start = datetime.now()

            try:
                # Generate
                outputs = model.generate(
                    input_ids=input_ids,
                    generation_config=gen_config
                )

                # Extract generated text
                generated_tokens = outputs.sequences[0]
                new_tokens = generated_tokens[input_length:]
                num_new_tokens = len(new_tokens)

                generated_text = tokenizer.decode(new_tokens, skip_special_tokens=True)

                # Check minimum tokens
                if num_new_tokens < args.min_tokens:
                    warnings.warn(
                        f"Trial {trial_id}: Only {num_new_tokens} tokens generated "
                        f"(minimum: {args.min_tokens}). Results may be unreliable."
                    )

                # Extract hidden states
                hidden_states = extract_hidden_states_from_generation(
                    outputs,
                    input_length,
                    args.layer
                )

                # Save raw if requested
                if args.save_raw:
                    raw_hidden_states[f'trial_{trial_id}'] = hidden_states

                # Create segments
                segments = create_overlapping_segments(
                    hidden_states,
                    args.n_segments,
                    args.overlap,
                    args.window_func
                )

                # Store segments with metadata
                for seg_idx in range(args.n_segments):
                    seg_position = seg_idx / (args.n_segments - 1) if args.n_segments > 1 else 0.0
                    segment_row = {
                        'trial_id': trial_id,
                        'segment_index': seg_idx,
                        'segment_position': seg_position
                    }
                    # Add dimensions
                    for dim_idx in range(hidden_dim):
                        segment_row[f'dim_{dim_idx}'] = segments[seg_idx, dim_idx]

                    all_segments.append(segment_row)

                # Compute metadata
                seg_meta = compute_segment_metadata(
                    num_new_tokens,
                    args.n_segments,
                    args.overlap
                )

                trial_meta = {
                    'trial_id': trial_id,
                    'generated_text': generated_text,
                    'original_num_tokens': num_new_tokens,
                    'effective_window_tokens': seg_meta['effective_window_tokens'],
                    'effective_hop_tokens': seg_meta['effective_hop_tokens'],
                    'timestamp': trial_start.isoformat()
                }
                trial_metadata.append(trial_meta)

                # Log generation
                generations_log.append(
                    f"=== Trial {trial_id} ({num_new_tokens} tokens) ===\n{generated_text}\n"
                )

                # Clean up
                del outputs, generated_tokens, hidden_states, segments
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            except Exception as e:
                print(f"\nError in trial {trial_id}: {e}")
                skipped_trials += 1
                continue

    timestamp_end = datetime.now()

    # Save segments.csv
    print("\nSaving segments.csv...")
    segments_df = pd.DataFrame(all_segments)
    segments_df.to_csv(output_dir / "segments.csv", index=False)

    # Save metadata.json
    print("Saving metadata.json...")
    token_counts = [t['original_num_tokens'] for t in trial_metadata]
    metadata = {
        "experiment": {
            "timestamp": timestamp_start.isoformat(),
            "script_version": "1.0.0"
        },
        "model": {
            "name": args.model,
            "hidden_dim": int(hidden_dim),
            "num_layers": int(num_layers),
            "extracted_layer": args.layer
        },
        "generation": {
            "prompt": args.prompt,
            "temperature": args.temperature,
            "max_new_tokens": args.max_new_tokens,
            "n_trials": args.n_trials
        },
        "segmentation": {
            "n_segments": args.n_segments,
            "overlap": args.overlap,
            "window_func": args.window_func,
            "min_tokens": args.min_tokens
        },
        "trials": trial_metadata,
        "statistics": {
            "total_trials": args.n_trials,
            "successful_trials": len(trial_metadata),
            "skipped_trials": skipped_trials,
            "token_counts": {
                "min": int(np.min(token_counts)) if token_counts else 0,
                "max": int(np.max(token_counts)) if token_counts else 0,
                "mean": float(np.mean(token_counts)) if token_counts else 0,
                "std": float(np.std(token_counts)) if token_counts else 0
            },
            "duration_seconds": (timestamp_end - timestamp_start).total_seconds()
        }
    }

    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    # Save generations.txt
    print("Saving generations.txt...")
    with open(output_dir / "generations.txt", 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("GENERATION LOG\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Prompt: \"{args.prompt}\"\n")
        f.write("=" * 80 + "\n\n")
        f.write("\n".join(generations_log))

    # Save raw hidden states if requested
    if args.save_raw:
        print("Saving raw_hidden_states.npz...")
        np.savez_compressed(output_dir / "raw_hidden_states.npz", **raw_hidden_states)

    # Validation
    print("\nValidating output...")
    validate_output(output_dir, args.n_segments, len(trial_metadata))

    # Summary
    print("\n" + "="*60)
    print("実行完了")
    print("="*60)
    print(f"出力ディレクトリ: {output_dir}")
    print(f"セグメントファイル: {output_dir}/segments.csv")
    print(f"  - 形状: ({len(trial_metadata) * args.n_segments}, {hidden_dim + 3})")
    print(f"  - 試行数: {len(trial_metadata)}")
    print(f"  - セグメント/試行: {args.n_segments}")
    print(f"\nトークン統計:")
    print(f"  - 最小: {metadata['statistics']['token_counts']['min']}")
    print(f"  - 最大: {metadata['statistics']['token_counts']['max']}")
    print(f"  - 平均: {metadata['statistics']['token_counts']['mean']:.1f}")
    print(f"  - 標準偏差: {metadata['statistics']['token_counts']['std']:.1f}")

    # Preview first trial
    print(f"\n最初の試行のプレビュー:")
    preview_df = segments_df[segments_df['trial_id'] == 0][['segment_index', 'segment_position']]
    print(preview_df.to_string(index=False))

    print("\n後続分析用コマンド例:")
    print(f"  python analyze_persistence.py --input {output_dir}/segments.csv")

    return metadata


def validate_output(output_dir: Path, n_segments: int, n_trials: int):
    """Validate output consistency."""

    # 1. Validate segments.csv
    df = pd.read_csv(output_dir / "segments.csv")

    # All trials have same number of segments
    segments_per_trial = df.groupby('trial_id').size()
    assert segments_per_trial.nunique() == 1, "Inconsistent segment counts"
    assert segments_per_trial.iloc[0] == n_segments, f"Expected {n_segments} segments"

    # segment_position in range [0, 1]
    assert df['segment_position'].min() >= 0, "segment_position < 0"
    assert df['segment_position'].max() <= 1, "segment_position > 1"

    # 2. Validate metadata.json
    with open(output_dir / "metadata.json") as f:
        meta = json.load(f)

    assert len(meta['trials']) == n_trials, "Inconsistent trial count"

    print("✓ 検証完了:")
    print(f"  - {n_trials} 試行")
    print(f"  - 各試行 {n_segments} セグメント")
    print(f"  - Hidden dim: {len([c for c in df.columns if c.startswith('dim_')])}")
    print(f"  - 総行数: {len(df)}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract LLM hidden states with STFT-style segmentation"
    )

    # Required
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Input prompt"
    )

    # Optional
    parser.add_argument(
        "--n_trials",
        type=int,
        default=10,
        help="Number of trials (default: 10)"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="microsoft/DialoGPT-medium",
        help="HuggingFace model name (default: microsoft/DialoGPT-medium)"
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature (default: 0.7)"
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=50,
        help="Maximum new tokens (default: 50)"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Output directory (default: ./output)"
    )

    parser.add_argument(
        "--layer",
        type=int,
        default=-1,
        help="Layer to extract, -1 for last layer (default: -1)"
    )

    parser.add_argument(
        "--n_segments",
        type=int,
        default=10,
        help="Number of segments (default: 10)"
    )

    parser.add_argument(
        "--overlap",
        type=float,
        default=0.5,
        help="Overlap ratio 0.0-0.9 (default: 0.5)"
    )

    parser.add_argument(
        "--window_func",
        type=str,
        default="hann",
        choices=["rect", "hann", "hamming"],
        help="Window function (default: hann)"
    )

    parser.add_argument(
        "--save_raw",
        action="store_true",
        help="Save raw hidden states"
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use float16 for model"
    )

    parser.add_argument(
        "--min_tokens",
        type=int,
        default=5,
        help="Minimum tokens threshold (default: 5)"
    )

    args = parser.parse_args()

    # Validate arguments
    if not 0.0 <= args.overlap <= 0.9:
        parser.error("overlap must be between 0.0 and 0.9")

    if args.n_segments < 1:
        parser.error("n_segments must be at least 1")

    # Run extraction
    run_extraction(args)


if __name__ == "__main__":
    main()

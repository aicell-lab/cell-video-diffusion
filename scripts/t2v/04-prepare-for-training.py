#!/usr/bin/env python3

"""
04-prepare-for-training.py

Takes the CSV with video paths and prompts, then creates the files needed for training:
- prompts.txt
- videos.txt
for training, validation, and test subsets.

Usage example:
  python 04-prepare-for-training.py \
    --input_csv ./output/extreme_phenotypes_with_prompts.csv \
    --output_dir ../../data/ready/IDR0013-FILTERED \
    --val_samples 2 \
    --test_percentage 10 \
    --prompt_type visual \
    --prompt_prefix "<T2V>"
"""

import argparse
import os
import pandas as pd
import random
import math

def main():
    parser = argparse.ArgumentParser(description="Prepare prompts and video paths for T2V training")
    parser.add_argument("--input_csv", type=str, 
                        default="/proj/aicell/users/x_aleho/video-diffusion/scripts/t2v/output/extreme_phenotypes_with_prompts.csv",
                        help="CSV containing video paths and prompts")
    parser.add_argument("--output_dir", type=str, default="./IDR0013-FILTERED",
                        help="Base output dir for train/validation/test subdirs")
    parser.add_argument("--val_samples", type=int, default=2,
                        help="Number of videos for in-training validation. Default=2")
    parser.add_argument("--test_percentage", type=int, default=10,
                        help="Percentage of videos to use for final test set. Default=10")
    parser.add_argument("--prompt_type", type=str, default="visual", choices=["technical", "visual"],
                        help="Which prompt type to use: 'technical' or 'visual'")
    parser.add_argument("--prompt_prefix", type=str, default="<TIMELAPSE>",
                        help="Prefix to add to the beginning of each prompt")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    # Define output directories
    train_dir = args.output_dir
    val_dir = f"{args.output_dir}-Val"
    test_dir = f"{args.output_dir}-Test"

    # Create output directories
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Load the CSV
    if not os.path.exists(args.input_csv):
        print(f"Error: Input CSV not found: {args.input_csv}")
        return

    df = pd.read_csv(args.input_csv)
    print(f"Loaded {len(df)} samples from {args.input_csv}")

    # Determine which prompt column to use
    prompt_column = f"{args.prompt_type}_prompt"
    if prompt_column not in df.columns:
        print(f"Error: Prompt column '{prompt_column}' not found in CSV")
        return

    # Create lists of prompts and video paths
    # Add prefix to prompts if specified
    if args.prompt_prefix:
        prompts = [f"{args.prompt_prefix} {p}" for p in df[prompt_column]]
    else:
        prompts = df[prompt_column].tolist()
    
    videos = df['video_path'].tolist()

    # Determine test set size (based on percentage)
    total_samples = len(df)
    test_size = math.ceil(total_samples * args.test_percentage / 100)
    
    # Determine validation set size
    val_size = min(args.val_samples, total_samples - test_size)
    
    # Calculate training set size
    train_size = total_samples - test_size - val_size

    print(f"Splitting data:")
    print(f"  - Training:   {train_size} samples ({train_size/total_samples:.1%})")
    print(f"  - Validation: {val_size} samples (for in-training validation)")
    print(f"  - Test:       {test_size} samples ({test_size/total_samples:.1%}) for final evaluation")

    # Create pairs of (prompt, video) for easier shuffling
    pairs = list(zip(prompts, videos))
    
    # Shuffle the data
    random.shuffle(pairs)
    
    # Split into training, validation and test sets
    test_pairs = pairs[:test_size]
    val_pairs = pairs[test_size:test_size+val_size]
    train_pairs = pairs[test_size+val_size:]
    
    # Unzip the pairs back into separate lists
    train_prompts, train_videos = zip(*train_pairs) if train_pairs else ([], [])
    val_prompts, val_videos = zip(*val_pairs) if val_pairs else ([], [])
    test_prompts, test_videos = zip(*test_pairs) if test_pairs else ([], [])

    # Write training files
    train_prompts_path = os.path.join(train_dir, "prompts.txt")
    train_videos_path = os.path.join(train_dir, "videos.txt")
    
    with open(train_prompts_path, "w", encoding="utf-8") as f:
        for prompt in train_prompts:
            f.write(f"{prompt}\n")
            
    with open(train_videos_path, "w", encoding="utf-8") as f:
        for video in train_videos:
            f.write(f"{video}\n")
    
    # Write validation files
    val_prompts_path = os.path.join(val_dir, "prompts.txt")
    val_videos_path = os.path.join(val_dir, "videos.txt")
    
    with open(val_prompts_path, "w", encoding="utf-8") as f:
        for prompt in val_prompts:
            f.write(f"{prompt}\n")
            
    with open(val_videos_path, "w", encoding="utf-8") as f:
        for video in val_videos:
            f.write(f"{video}\n")
    
    # Write test files
    test_prompts_path = os.path.join(test_dir, "prompts.txt")
    test_videos_path = os.path.join(test_dir, "videos.txt")
    
    with open(test_prompts_path, "w", encoding="utf-8") as f:
        for prompt in test_prompts:
            f.write(f"{prompt}\n")
            
    with open(test_videos_path, "w", encoding="utf-8") as f:
        for video in test_videos:
            f.write(f"{video}\n")
    
    print(f"\nWrote training set => {len(train_prompts)} lines:")
    print(f"  {train_prompts_path}")
    print(f"  {train_videos_path}")
    
    print(f"\nWrote validation set => {len(val_prompts)} lines:")
    print(f"  {val_prompts_path}")
    print(f"  {val_videos_path}")
    
    print(f"\nWrote test set => {len(test_prompts)} lines:")
    print(f"  {test_prompts_path}")
    print(f"  {test_videos_path}")
    
    # Print a few examples of the prompts
    if train_prompts:
        print("\nSample training prompts:")
        for i in range(min(3, len(train_prompts))):
            print(f"  {train_prompts[i]}")
    
    if val_prompts:
        print("\nSample validation prompts:")
        for i in range(min(2, len(val_prompts))):
            print(f"  {val_prompts[i]}")
    
    if test_prompts:
        print("\nSample test prompts:")
        for i in range(min(3, len(test_prompts))):
            print(f"  {test_prompts[i]}")

if __name__ == "__main__":
    main() 
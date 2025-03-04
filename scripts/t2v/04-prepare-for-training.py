#!/usr/bin/env python3

"""
04-prepare-for-training.py

Takes the CSV with video paths and prompts, then creates the files needed for training:
- prompts.txt
- videos.txt
- phenotypes.csv (when using phenotype conditioning)
for training, validation, and test subsets.

Usage example:
  python 04-prepare-for-training.py \
    --input_csv ./output/extreme_phenotypes_with_prompts.csv \
    --output_dir ../../data/ready/IDR0013-FILTERED \
    --val_samples 2 \
    --test_percentage 10 \
    --prompt_type visual \
    --prompt_prefix "<T2V>"

  # For phenotype conditioning:
  python 04-prepare-for-training.py \
    --input_csv ./output/extreme_phenotypes_with_prompts.csv \
    --output_dir ../../data/ready/IDR0013-FILTERED-2 \
    --val_samples 2 \
    --test_percentage 10 \
    --conditioning_type phenotype \
    --phenotype_columns proliferation_score_normalized,migration_speed_score_normalized,cell_death_score_normalized \
    --base_prompt "Time-lapse microscopy video of multiple cells."
"""

import argparse
import os
import pandas as pd
import random
import math
import csv

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
    
    # Add conditioning type argument
    parser.add_argument("--conditioning_type", type=str, default="text", choices=["text", "phenotype"],
                        help="Type of conditioning to use: 'text' (uses prompts) or 'phenotype' (uses numerical values)")
    
    # Arguments for text conditioning
    parser.add_argument("--prompt_type", type=str, default="visual", choices=["technical", "visual"],
                        help="Which prompt type to use (for text conditioning): 'technical' or 'visual'")
    parser.add_argument("--prompt_prefix", type=str, default="<T2V>",
                        help="Prefix to add to the beginning of each prompt (for text conditioning)")
    
    # Arguments for phenotype conditioning
    parser.add_argument("--phenotype_columns", type=str, 
                        default="proliferation_score_normalized,migration_speed_score_normalized,cell_death_score_normalized",
                        help="Comma-separated list of phenotype columns to use for conditioning")
    parser.add_argument("--base_prompt", type=str, default="Time-lapse microscopy video of multiple cells.",
                        help="Base text prompt to use with phenotype conditioning")
    
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
    
    # Handle different conditioning types
    if args.conditioning_type == "text":
        # Text-based conditioning (original approach)
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
            
    elif args.conditioning_type == "phenotype":
        # Phenotype-based conditioning (new approach)
        # Use the same base prompt for all samples
        if args.prompt_prefix:
            prompts = [f"{args.prompt_prefix} {args.base_prompt}" for _ in range(len(df))]
        else:
            prompts = [args.base_prompt for _ in range(len(df))]
            
        # Parse phenotype columns
        phenotype_cols = args.phenotype_columns.split(',')
        print(f"Using phenotype columns: {phenotype_cols}")
        
        # Check if all phenotype columns exist
        for col in phenotype_cols:
            if col not in df.columns:
                print(f"Error: Phenotype column '{col}' not found in CSV")
                return
                
        # Extract phenotype values
        phenotypes = df[phenotype_cols].values
        print(f"Phenotype shape: {phenotypes.shape}")
        
        # Create simple column names for the CSV
        phenotype_simple_names = []
        for col in phenotype_cols:
            if "proliferation" in col:
                phenotype_simple_names.append("proliferation")
            elif "migration" in col:
                phenotype_simple_names.append("migration_speed")
            elif "cell_death" in col:
                phenotype_simple_names.append("cell_death")
            else:
                # Use original name if no match
                phenotype_simple_names.append(col)
    
    # Get video paths (common for both approaches)    
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

    # Create pairs/triplets of data for shuffling
    if args.conditioning_type == "text":
        # For text conditioning: (prompt, video)
        data_pairs = list(zip(prompts, videos))
        
        # Shuffle the data
        random.shuffle(data_pairs)
        
        # Split into training, validation and test sets
        test_pairs = data_pairs[:test_size]
        val_pairs = data_pairs[test_size:test_size+val_size]
        train_pairs = data_pairs[test_size+val_size:]
        
        # Unzip the pairs back into separate lists
        train_prompts, train_videos = zip(*train_pairs) if train_pairs else ([], [])
        val_prompts, val_videos = zip(*val_pairs) if val_pairs else ([], [])
        test_prompts, test_videos = zip(*test_pairs) if test_pairs else ([], [])
        
    else:  # phenotype conditioning
        # For phenotype conditioning: (prompt, video, phenotype)
        data_triplets = list(zip(prompts, videos, [p for p in phenotypes]))
        
        # Shuffle the data
        random.shuffle(data_triplets)
        
        # Split into training, validation and test sets
        test_triplets = data_triplets[:test_size]
        val_triplets = data_triplets[test_size:test_size+val_size]
        train_triplets = data_triplets[test_size+val_size:]
        
        # Unzip the triplets back into separate lists
        train_prompts, train_videos, train_phenotypes = zip(*train_triplets) if train_triplets else ([], [], [])
        val_prompts, val_videos, val_phenotypes = zip(*val_triplets) if val_triplets else ([], [], [])
        test_prompts, test_videos, test_phenotypes = zip(*test_triplets) if test_triplets else ([], [], [])

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
            
    # Write phenotype files (if using phenotype conditioning)
    if args.conditioning_type == "phenotype":
        train_phenotypes_path = os.path.join(train_dir, "phenotypes.csv")
        val_phenotypes_path = os.path.join(val_dir, "phenotypes.csv")
        test_phenotypes_path = os.path.join(test_dir, "phenotypes.csv")
        
        # Write training phenotypes
        with open(train_phenotypes_path, "w", encoding="utf-8", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(phenotype_simple_names)  # Header row
            for phenotype in train_phenotypes:
                writer.writerow(phenotype)
                
        # Write validation phenotypes
        with open(val_phenotypes_path, "w", encoding="utf-8", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(phenotype_simple_names)  # Header row
            for phenotype in val_phenotypes:
                writer.writerow(phenotype)
                
        # Write test phenotypes
        with open(test_phenotypes_path, "w", encoding="utf-8", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(phenotype_simple_names)  # Header row
            for phenotype in test_phenotypes:
                writer.writerow(phenotype)
    
    print(f"\nWrote training set => {len(train_prompts)} lines:")
    print(f"  {train_prompts_path}")
    print(f"  {train_videos_path}")
    if args.conditioning_type == "phenotype":
        print(f"  {train_phenotypes_path}")
    
    print(f"\nWrote validation set => {len(val_prompts)} lines:")
    print(f"  {val_prompts_path}")
    print(f"  {val_videos_path}")
    if args.conditioning_type == "phenotype":
        print(f"  {val_phenotypes_path}")
    
    print(f"\nWrote test set => {len(test_prompts)} lines:")
    print(f"  {test_prompts_path}")
    print(f"  {test_videos_path}")
    if args.conditioning_type == "phenotype":
        print(f"  {test_phenotypes_path}")
    
    # Print a few examples of the prompts
    if train_prompts:
        print("\nSample training prompts:")
        for i in range(min(3, len(train_prompts))):
            print(f"  {train_prompts[i]}")
            if args.conditioning_type == "phenotype":
                print(f"    Phenotypes: {','.join(map(str, train_phenotypes[i]))}")
    
    if val_prompts:
        print("\nSample validation prompts:")
        for i in range(min(2, len(val_prompts))):
            print(f"  {val_prompts[i]}")
            if args.conditioning_type == "phenotype":
                print(f"    Phenotypes: {','.join(map(str, val_phenotypes[i]))}")
    
    if test_prompts:
        print("\nSample test prompts:")
        for i in range(min(3, len(test_prompts))):
            print(f"  {test_prompts[i]}")
            if args.conditioning_type == "phenotype":
                print(f"    Phenotypes: {','.join(map(str, test_phenotypes[i]))}")

if __name__ == "__main__":
    main() 
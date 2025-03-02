#!/usr/bin/env python3

"""
06-prepare_dataset.py

Takes the visual prompts CSV file and creates the dataset files needed for training:
  1) Reads the visual prompts CSV that contains Gene Symbol and visual_prompt
  2) Reads the annotation CSV that maps Plate and Well Number to Gene Symbol
  3) Reads the video paths CSV that maps plate_well_id to video_path
  4) Merges these datasets to create a mapping from Gene Symbol to video_path
  5) Writes out two sets of matching lines: 
     - prompts.txt
     - videos.txt 
     for training and validation subsets.
  6) Prompts look like:
     "<ALEXANDER> {visual_prompt}"

Usage example:
  python scripts/06-prepare_dataset.py \
    --visual_prompts_csv scripts/visual_prompts.csv \
    --annotation_csv data/processed/idr0013/idr0013-screenA-annotation.csv \
    --video_paths_csv scripts/video_paths.csv \
    --output_dir data/ready/gene-knockdown-visual \
    --val_fraction 0.1 \
    --prompt_prefix "<ALEXANDER>"
"""

import argparse
import csv
import os
import random
import pandas as pd
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Prepare dataset from visual prompts")
    parser.add_argument("--visual_prompts_csv", type=str, default="visual_prompts.csv",
                        help="CSV containing Gene Symbol and visual_prompt")
    parser.add_argument("--annotation_csv", type=str, 
                        default="data/processed/idr0013/idr0013-screenA-annotation.csv",
                        help="CSV containing Plate, Well Number, and Gene Symbol mapping")
    parser.add_argument("--video_paths_csv", type=str, default="scripts/video_paths.csv",
                        help="CSV containing plate_well_id and video_path mapping")
    parser.add_argument("--output_dir", type=str, default="data/ready/gene-knockdown-visual",
                        help="Base output dir for train/validation subdirs")
    parser.add_argument("--val_fraction", type=float, default=0.1,
                        help="Fraction of data to use for validation (0-1)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for train/val split")
    parser.add_argument("--prompt_prefix", type=str, default="<ALEXANDER>",
                        help="Prefix to add to the beginning of each prompt")
    args = parser.parse_args()

    # Set random seed for reproducibility
    random.seed(args.seed)

    # Create output directories
    train_dir = args.output_dir
    val_dir = f"{args.output_dir}-Val"
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Load the visual prompts
    print(f"Loading visual prompts from {args.visual_prompts_csv}")
    visual_prompts_df = pd.read_csv(args.visual_prompts_csv)
    
    # Load the annotation data
    print(f"Loading annotation data from {args.annotation_csv}")
    annotation_df = pd.read_csv(args.annotation_csv)
    
    # Load the video paths
    print(f"Loading video paths from {args.video_paths_csv}")
    video_paths_df = pd.read_csv(args.video_paths_csv)
    
    # Create a plate_well_id column in the annotation dataframe
    annotation_df['plate_well_id'] = annotation_df['Plate'] + '-' + annotation_df['Well Number'].astype(str).str.zfill(5) + '_01'
    
    # Merge annotation with video paths on plate_well_id
    print("Merging annotation with video paths")
    annotation_video_df = pd.merge(annotation_df, video_paths_df, on="plate_well_id", how="inner")
    
    # Select only the columns we need
    annotation_video_df = annotation_video_df[['Gene Symbol', 'video_path']]
    
    # Merge with visual prompts on Gene Symbol
    print("Merging with visual prompts")
    merged_df = pd.merge(annotation_video_df, visual_prompts_df, on="Gene Symbol", how="inner")
    
    # Filter out rows with missing visual prompts
    filtered_df = merged_df.dropna(subset=["visual_prompt"])
    
    # Count before and after filtering
    total_genes = len(visual_prompts_df['Gene Symbol'].unique())
    total_videos = len(annotation_video_df)
    total_merged = len(merged_df)
    total_filtered = len(filtered_df)
    
    print(f"Total unique genes in visual prompts file: {total_genes}")
    print(f"Total videos with gene symbols: {total_videos}")
    print(f"Total merged entries: {total_merged}")
    print(f"Total entries with valid visual prompts: {total_filtered}")
    
    if total_filtered == 0:
        print("No valid data found. Exiting.")
        return
    
    # Prepare the data for train/val split
    data = []
    for _, row in filtered_df.iterrows():
        gene = row["Gene Symbol"]
        visual_prompt = row["visual_prompt"]
        video_path = row["video_path"]
        
        # Format the prompt with the prefix
        formatted_prompt = f"{args.prompt_prefix} {visual_prompt}"
        
        data.append({
            "gene": gene,
            "prompt": formatted_prompt,
            "video_path": video_path
        })
    
    # Group by gene to avoid having the same gene in both train and validation sets
    gene_groups = {}
    for item in data:
        gene = item["gene"]
        if gene not in gene_groups:
            gene_groups[gene] = []
        gene_groups[gene].append(item)
    
    # Get list of genes and shuffle
    genes = list(gene_groups.keys())
    random.shuffle(genes)
    
    # Split genes into train and validation sets
    val_genes_count = max(1, int(len(genes) * args.val_fraction))
    val_genes = genes[:val_genes_count]
    train_genes = genes[val_genes_count:]
    
    # Create train and validation datasets
    train_data = []
    for gene in train_genes:
        train_data.extend(gene_groups[gene])
    
    val_data = []
    for gene in val_genes:
        val_data.extend(gene_groups[gene])
    
    # Shuffle the data within each set
    random.shuffle(train_data)
    random.shuffle(val_data)
    
    print(f"Training genes: {len(train_genes)}, samples: {len(train_data)}")
    print(f"Validation genes: {len(val_genes)}, samples: {len(val_data)}")
    
    # Write out the training files
    train_prompts_path = os.path.join(train_dir, "prompts.txt")
    train_videos_path = os.path.join(train_dir, "videos.txt")
    
    with open(train_prompts_path, "w", encoding="utf-8") as f_p, open(train_videos_path, "w", encoding="utf-8") as f_v:
        for item in train_data:
            f_p.write(item["prompt"] + "\n")
            f_v.write(item["video_path"] + "\n")
    
    # Write out the validation files
    val_prompts_path = os.path.join(val_dir, "prompts.txt")
    val_videos_path = os.path.join(val_dir, "videos.txt")
    
    with open(val_prompts_path, "w", encoding="utf-8") as f_p, open(val_videos_path, "w", encoding="utf-8") as f_v:
        for item in val_data:
            f_p.write(item["prompt"] + "\n")
            f_v.write(item["video_path"] + "\n")
    
    print(f"Wrote training set => {len(train_data)} lines:\n  {train_prompts_path}\n  {train_videos_path}")
    print(f"Wrote validation set => {len(val_data)} lines:\n  {val_prompts_path}\n  {val_videos_path}")
    
    # Write a sample of the prompts for inspection
    sample_size = min(5, len(train_data))
    print(f"\nSample of {sample_size} training prompts:")
    for i in range(sample_size):
        print(f"Gene: {train_data[i]['gene']}")
        print(f"Prompt: {train_data[i]['prompt']}")
        print(f"Video: {train_data[i]['video_path']}")
        print()

if __name__ == "__main__":
    main()

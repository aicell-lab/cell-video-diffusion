#!/usr/bin/env python3

import os
import pandas as pd
import argparse
import numpy as np

# Path to the previous CSV with video paths and cell counts
INPUT_CSV_PATH = "/proj/aicell/users/x_aleho/video-diffusion/scripts/t2v/output/extreme_phenotypes_with_cell_counts.csv"

# Output path for the new CSV with prompts
OUTPUT_CSV_PATH = "/proj/aicell/users/x_aleho/video-diffusion/scripts/t2v/output/extreme_phenotypes_with_prompts.csv"

# Mappings for visual prompts
VISUAL_INITIAL_COUNT_MAP = {
    "HIGH": "many cells",
    "MED": "several cells",
    "LOW": "a few cells",
}

VISUAL_PROLIFERATION_MAP = {
    "HIGH": "divide often",
    "MED": "occasionally divide",
    "LOW": "rarely divide",
}

VISUAL_MIGRATION_MAP = {
    "HIGH": "move rapidly",
    "MED": "move moderately",
    "LOW": "move slowly",
}

VISUAL_CELL_DEATH_MAP = {
    "HIGH": "frequently disappear",
    "MED": "occasionally disappear",
    "LOW": "rarely disappear",
}

def create_visual_prompt(row):
    """
    Create a visual prompt from phenotype labels.
    Format: "Time-lapse microscopy video of [INITIAL_COUNT]. The cells [PROLIFERATION], 
    [MIGRATION], and [CELL_DEATH] due to cell death."
    """
    # Start with the base description, including cell count if available
    if 'initial_cell_count_label' in row and pd.notna(row['initial_cell_count_label']):
        base = f"Time-lapse microscopy video of {VISUAL_INITIAL_COUNT_MAP[row['initial_cell_count_label']]}"
    else:
        base = "Time-lapse microscopy video of multiple cells"
    
    # Add behavioral descriptions
    visual_parts = []
    visual_parts.append(VISUAL_PROLIFERATION_MAP[row['proliferation_label']])
    visual_parts.append(VISUAL_MIGRATION_MAP[row['migration_speed_label']])
    visual_parts.append(f"{VISUAL_CELL_DEATH_MAP[row['cell_death_label']]} due to cell death")
    
    # Construct the prompt with the new format
    return f"{base}. The cells {', '.join(visual_parts[:-1])}, and {visual_parts[-1]}."

def normalize_phenotype_scores(df):
    """
    Normalize phenotype scores to a range of [0, 1] for better model conditioning.
    Creates new columns with normalized scores.
    """
    score_columns = [
        'initial_cell_count', 
        'proliferation_score', 
        'migration_speed_score', 
        'cell_death_score'
    ]
    
    # Create normalized versions of each score
    for col in score_columns:
        if col in df.columns:
            min_val = df[col].min()
            max_val = df[col].max()
            norm_col = f"{col}_normalized"
            df[norm_col] = (df[col] - min_val) / (max_val - min_val)
            
    return df

def main():
    parser = argparse.ArgumentParser(description="Create visual prompts for time-lapse videos")
    parser.add_argument('--input', type=str, default=INPUT_CSV_PATH, 
                        help='Path to input CSV with video paths')
    parser.add_argument('--output', type=str, default=OUTPUT_CSV_PATH,
                        help='Path to output CSV with prompts')
    parser.add_argument('--normalize_scores', action='store_true',
                        help='Normalize phenotype scores to [0,1] range')
    args = parser.parse_args()
    
    print(f"Loading data from {args.input}")
    
    # Load the CSV with video information
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return
    
    # Load data
    df = pd.read_csv(args.input)
    print(f"Loaded {len(df)} samples")
    
    # Generate prompts
    print("Generating visual prompts...")
    df['prompt'] = df.apply(create_visual_prompt, axis=1)
    
    # Normalize phenotype scores if requested
    if args.normalize_scores:
        print("Normalizing phenotype scores...")
        df = normalize_phenotype_scores(df)
    
    # Save the enhanced CSV
    print(f"Saving enhanced data to {args.output}")
    df.to_csv(args.output, index=False)
    
    # Print example prompts and scores
    print("\nExample prompts and scores:")
    for i in range(min(5, len(df))):
        print(f"\nSample {i+1}:")
        print(f"- Prompt: {df.iloc[i]['prompt']}")
        
        # Print score information if available
        score_cols = [col for col in df.columns if 'score' in col or col in ['initial_cell_count', 'initial_cell_count_normalized']]
        if score_cols:
            print("- Phenotype scores:")
            for col in score_cols:
                if col in df.iloc[i]:
                    print(f"  {col}: {df.iloc[i][col]:.4f}")
    
    print(f"\nComplete! Generated {len(df)} prompts")

if __name__ == "__main__":
    main() 
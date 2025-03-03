#!/usr/bin/env python3

import os
import pandas as pd
import argparse

# Path to the previous CSV with video paths
INPUT_CSV_PATH = "/proj/aicell/users/x_aleho/video-diffusion/scripts/t2v/output/extreme_phenotypes_with_videos.csv"

# Output path for the new CSV with prompts
OUTPUT_CSV_PATH = "/proj/aicell/users/x_aleho/video-diffusion/scripts/t2v/output/extreme_phenotypes_with_prompts.csv"

# Mappings for technical prompts
TECH_PROLIFERATION_MAP = {
    "HIGH": "HIGH proliferation",
    "MED": "MEDIUM proliferation",
    "LOW": "LOW proliferation",
}

TECH_MIGRATION_MAP = {
    "HIGH": "FAST migration",
    "MED": "MEDIUM migration",
    "LOW": "SLOW migration",
}

TECH_CELL_DEATH_MAP = {
    "HIGH": "HIGH cell death",
    "MED": "MEDIUM cell death",
    "LOW": "LOW cell death",
}

# Mappings for visual prompts
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

def create_technical_prompt(row):
    """
    Create a technical prompt from phenotype labels.
    Format: "HeLa time-lapse with [PROLIFERATION], [MIGRATION], and [CELL DEATH]."
    """
    elements = []
    
    # Add all available phenotypes
    elements.append(TECH_PROLIFERATION_MAP[row['proliferation_label']])
    elements.append(TECH_MIGRATION_MAP[row['migration_speed_label']])
    elements.append(TECH_CELL_DEATH_MAP[row['cell_death_label']])
    
    # Construct the prompt with proper grammar
    return f"HeLa time-lapse with {', '.join(elements[:-1])}, and {elements[-1]}."

def create_visual_prompt(row):
    """
    Create a visual prompt from phenotype labels.
    Format: "Time-lapse microscopy video of multiple cells. The cells [PROLIFERATION], 
    [MIGRATION], and [CELL DEATH] due to cell death."
    """
    visual_parts = []
    
    # Add all available phenotypes
    visual_parts.append(VISUAL_PROLIFERATION_MAP[row['proliferation_label']])
    visual_parts.append(VISUAL_MIGRATION_MAP[row['migration_speed_label']])
    visual_parts.append(f"{VISUAL_CELL_DEATH_MAP[row['cell_death_label']]} due to cell death")
    
    # Construct the prompt with the new format
    return f"Time-lapse microscopy video of multiple cells. The cells {', '.join(visual_parts[:-1])}, and {visual_parts[-1]}."

def main():
    parser = argparse.ArgumentParser(description="Create technical and visual prompts for time-lapse videos")
    parser.add_argument('--input', type=str, default=INPUT_CSV_PATH, 
                        help='Path to input CSV with video paths')
    parser.add_argument('--output', type=str, default=OUTPUT_CSV_PATH,
                        help='Path to output CSV with prompts')
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
    print("Generating technical and visual prompts...")
    df['technical_prompt'] = df.apply(create_technical_prompt, axis=1)
    df['visual_prompt'] = df.apply(create_visual_prompt, axis=1)
    
    # Save the enhanced CSV
    print(f"Saving enhanced data to {args.output}")
    df.to_csv(args.output, index=False)
    
    # Print example prompts
    print("\nExample prompts:")
    for i in range(min(5, len(df))):
        print(f"\nSample {i+1}:")
        print(f"- Technical: {df.iloc[i]['technical_prompt']}")
        print(f"- Visual: {df.iloc[i]['visual_prompt']}")
    
    print(f"\nComplete! Generated {len(df)} prompts")

if __name__ == "__main__":
    main() 
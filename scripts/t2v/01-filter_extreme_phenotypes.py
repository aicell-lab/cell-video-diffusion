#!/usr/bin/env python3

"""
01-filter_extreme_phenotypes.py

Filters cell phenotype data to identify samples with extreme phenotype combinations.
Creates a filtered CSV file containing samples with at least 2 extreme phenotypes
(HIGH or LOW) across cell death, migration speed, and proliferation metrics.

Usage:
  python 01-filter_extreme_phenotypes.py
"""

import os
import pandas as pd
import numpy as np
import re
from pathlib import Path

# Path to the annotation file
ANNOTATION_PATH = "/proj/aicell/users/x_aleho/video-diffusion/data/processed/idr0013/idr0013-screenA-annotation.csv"

# Output directory for the filtered data
OUTPUT_DIR = "/proj/aicell/users/x_aleho/video-diffusion/scripts/t2v/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_annotations():
    """Load and clean the annotation data."""
    print(f"Loading annotations from {ANNOTATION_PATH}")
    df = pd.read_csv(ANNOTATION_PATH)
    
    # Extract relevant columns
    columns_of_interest = [
        'Plate', 'Well', 'Plate_Well',
        'Well Number', # This should correspond to the CH5 filename (without extension)
        'Score - cell death (automatic)',
        'Score - migration (speed) (automatic)',
        'Score - increased proliferation (automatic)',
        'Has Phenotype'  # Including this for potential filtering
    ]
    
    # Keep only specified columns
    filtered_df = df[columns_of_interest].copy()
    
    # Rename columns for easier reference
    filtered_df = filtered_df.rename(columns={
        'Score - cell death (automatic)': 'cell_death',
        'Score - migration (speed) (automatic)': 'migration_speed',
        'Score - increased proliferation (automatic)': 'proliferation',
        'Well Number': 'well_number'  # Standardize name
    })
    
    # Remove rows with NaN values in critical columns
    filtered_df = filtered_df.dropna(subset=['cell_death', 'migration_speed', 'proliferation'])
    
    print(f"Loaded {len(filtered_df)} valid samples with complete phenotype scores")
    return filtered_df


def assign_phenotype_labels(df, percentile_threshold=10):
    """
    Assign HIGH/LOW/MED labels based on percentile thresholds.
    
    Args:
        df: DataFrame with phenotype scores
        percentile_threshold: Percentile threshold for extreme values (default: 10)
                             Values below this percentile are LOW
                             Values above (100-percentile_threshold) are HIGH
                             Values in between are MED
    
    Returns:
        DataFrame with additional phenotype label columns
    """
    result_df = df.copy()
    
    # Define thresholds for each phenotype
    for phenotype in ['cell_death', 'migration_speed', 'proliferation']:
        low_threshold = np.percentile(df[phenotype], percentile_threshold)
        high_threshold = np.percentile(df[phenotype], 100 - percentile_threshold)
        
        # Create new column for phenotype label
        label_col = f"{phenotype}_label"
        result_df[label_col] = 'MED'
        result_df.loc[result_df[phenotype] <= low_threshold, label_col] = 'LOW'
        result_df.loc[result_df[phenotype] >= high_threshold, label_col] = 'HIGH'
        
        # Print thresholds for reference
        print(f"{phenotype}: LOW <= {low_threshold:.6f}, HIGH >= {high_threshold:.6f}")
    
    return result_df

def filter_for_extreme_combinations(df, min_extreme_count=2):
    """
    Filter for samples with extreme phenotype combinations.
    Returns samples that have at least min_extreme_count extreme (HIGH or LOW) phenotypes.
    """
    extreme_df = df.copy()
    
    # Count extreme phenotypes (HIGH or LOW) for each sample
    extreme_df['extreme_count'] = 0
    for phenotype in ['proliferation_label', 'migration_speed_label', 'cell_death_label']:
        extreme_df['extreme_count'] += ((extreme_df[phenotype] == 'HIGH') | 
                                        (extreme_df[phenotype] == 'LOW')).astype(int)
    
    # Keep samples with at least min_extreme_count extreme phenotypes
    filtered_extreme = extreme_df[extreme_df['extreme_count'] >= min_extreme_count]
    
    print(f"\nFiltered samples by extreme phenotype count ({min_extreme_count}+):")
    print(f"Original samples: {len(df)}")
    print(f"Filtered samples: {len(filtered_extreme)}")
    
    # Print distribution of extreme phenotypes
    for count in range(1, 4):
        num_samples = len(extreme_df[extreme_df['extreme_count'] == count])
        print(f"Samples with {count} extreme phenotypes: {num_samples}")
    
    return filtered_extreme

def main():
    # Load and clean annotations
    annotations_df = load_annotations()
    
    # Print summary of original data
    print("\nOriginal data summary:")
    for col in ['cell_death', 'migration_speed', 'proliferation']:
        print(f"{col}: min={annotations_df[col].min():.6f}, max={annotations_df[col].max():.6f}, mean={annotations_df[col].mean():.6f}")
    
    # Assign phenotype labels based on percentiles
    print("\nAssigning phenotype labels...")
    labeled_df = assign_phenotype_labels(annotations_df, percentile_threshold=10)
    
    # Filter for samples with extreme phenotype combinations
    filtered_df = filter_for_extreme_combinations(labeled_df, min_extreme_count=2)
    
    # Save filtered results
    output_path = os.path.join(OUTPUT_DIR, "filtered_extreme_phenotypes.csv")
    filtered_df.to_csv(output_path, index=False)
    print(f"\nSaved {len(filtered_df)} filtered samples to {output_path}")
    
    # Print distribution of phenotype combinations
    print("\nDistribution of phenotype combinations in filtered data:")
    for phenotype in ['proliferation_label', 'migration_speed_label', 'cell_death_label']:
        value_counts = filtered_df[phenotype].value_counts()
        print(f"\n{phenotype}:")
        for value, count in value_counts.items():
            print(f"  {value}: {count} samples ({count/len(filtered_df)*100:.1f}%)")

if __name__ == "__main__":
    main() 
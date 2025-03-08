import os
import shutil
import pandas as pd
from pathlib import Path

# Base directories
base_dir = "masks_output/t2v"
source_dir = f"{base_dir}/IDR0013-FILTERED-Test"

# Categories we want to create subdirectories for
categories = {
    'initial_cell_count_label': 'cc',
    'cell_death_label': 'cd',
    'migration_speed_label': 'ms',
    'proliferation_label': 'pr'
}

# For each category and label, create a subdirectory and copy appropriate mask files
for category, abbrev in categories.items():
    for label in ['LOW', 'MED', 'HIGH']:
        # Create the target directory
        target_dir = f"{base_dir}/IDR0013-FILTERED-Test-{abbrev}-{label}"
        os.makedirs(target_dir, exist_ok=True)
        print(f"Created directory: {target_dir}")
        
        # Read the corresponding subset file
        subset_file = f"subsets/videos_{category}_{label}.txt"
        if not os.path.exists(subset_file):
            print(f"Warning: Subset file {subset_file} not found. Skipping.")
            continue
            
        # Get the list of video paths in this subset
        with open(subset_file, 'r') as f:
            video_paths = [line.strip() for line in f.readlines()]
        
        # Extract the base names for matching with mask files
        video_basenames = [os.path.basename(path).replace('.mp4', '') for path in video_paths]
        
        # Copy matching mask files to the target directory
        mask_count = 0
        for mask_file in os.listdir(source_dir):
            if not mask_file.endswith('_masks.npy'):
                continue
                
            # Check if this mask belongs to a video in our subset
            for video_base in video_basenames:
                if video_base in mask_file:
                    source_path = os.path.join(source_dir, mask_file)
                    target_path = os.path.join(target_dir, mask_file)
                    shutil.copy2(source_path, target_path)
                    mask_count += 1
                    break
        
        print(f"Copied {mask_count} mask files to {target_dir}")

print("Organization complete!")
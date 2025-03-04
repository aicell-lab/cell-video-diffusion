#!/usr/bin/env python3

"""
organize_masks_by_phenotype.py

Creates copies of mask files organized by phenotype level.
For example, splits masks from a single directory into:
- masks_output/IDR0013-FILTERED-Test-pr-LOW
- masks_output/IDR0013-FILTERED-Test-pr-MED
- masks_output/IDR0013-FILTERED-Test-pr-HIGH

Usage:
  python organize_masks_by_phenotype.py \
    --input_dir masks_output/IDR0013-FILTERED-Test \
    --phenotype cd  # Options: pr (proliferation), ms (migration), cd (cell death)
"""

import os
import argparse
import shutil
import re
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Organize mask files by phenotype level")
    parser.add_argument("--input_dir", required=True, help="Directory containing mask files")
    parser.add_argument("--phenotype", required=True, choices=["pr", "ms", "cd"], 
                       help="Phenotype to organize by: pr (proliferation), ms (migration), cd (cell death)")
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}")
        return
    
    # Define phenotype pattern to search for
    pattern = f"{args.phenotype}-(LOW|MED|HIGH)"
    
    # Get list of mask files
    mask_files = [f for f in os.listdir(args.input_dir) if f.endswith(".npy")]
    print(f"Found {len(mask_files)} mask files in {args.input_dir}")
    
    # Create output directories
    base_dir = args.input_dir
    if base_dir.endswith('/'):
        base_dir = base_dir[:-1]
        
    # Dictionary to count files per level
    level_counts = {"LOW": 0, "MED": 0, "HIGH": 0, "unknown": 0}
    
    # Process each mask file
    for mask_file in tqdm(mask_files, desc="Organizing files"):
        # Extract phenotype level using regex
        match = re.search(pattern, mask_file)
        if match:
            level = match.group(1)  # LOW, MED, or HIGH
            # Create output directory if it doesn't exist
            output_dir = f"{base_dir}-{args.phenotype}-{level}"
            os.makedirs(output_dir, exist_ok=True)
            
            # Copy file to appropriate directory
            src_path = os.path.join(args.input_dir, mask_file)
            dst_path = os.path.join(output_dir, mask_file)
            if not os.path.exists(dst_path):
                shutil.copy2(src_path, dst_path)
                level_counts[level] += 1
        else:
            level_counts["unknown"] += 1
            print(f"Warning: Could not find pattern '{pattern}' in {mask_file}")
    
    # Print summary
    print("\nFiles copied to each directory:")
    for level, count in level_counts.items():
        if level != "unknown":
            output_dir = f"{base_dir}-{args.phenotype}-{level}"
            print(f"  {output_dir}: {count} files")
    
    if level_counts["unknown"] > 0:
        print(f"\nWarning: {level_counts['unknown']} files did not match pattern '{pattern}'")

if __name__ == "__main__":
    main() 
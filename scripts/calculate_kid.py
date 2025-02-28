#!/usr/bin/env python3
"""Calculate Kernel Inception Distance (KID) between two image directories.

Sample usage:
    python calculate_kid.py --real-path ../data/processed/idr0013/final_frames/ --gen-path ../data/generated/test_generations_plateval/i2v_r128_250/frame_final/
    
Advanced usage:
    python calculate_kid.py --real-path /path/to/real/images --gen-path /path/to/generated/images \
        --batch-size 64 --device cuda --kid-subsets 100 --kid-subset-size 1000 --no-balance
"""

import argparse
import os
import random
import shutil
import tempfile
from pathlib import Path

import torch
from torch_fidelity import calculate_metrics
from tqdm import tqdm


def count_images(directory):
    """Count the number of image files in a directory."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    count = 0
    for root, _, files in os.walk(directory):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                count += 1
    return count


def get_image_files(directory):
    """Get all image files in a directory."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
    image_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                image_files.append(os.path.join(root, file))
    return image_files


def check_directory(directory):
    """Check if directory exists and contains images."""
    if not os.path.isdir(directory):
        raise ValueError(f"Directory does not exist: {directory}")
    
    image_count = count_images(directory)
    if image_count == 0:
        raise ValueError(f"No images found in directory: {directory}")
    
    return image_count


def create_balanced_sample(source_dir, target_count):
    """Create a balanced sample of images from source_dir."""
    temp_dir = tempfile.mkdtemp()
    
    # Get all image files
    all_images = get_image_files(source_dir)
    
    # If we have fewer images than target, use all of them
    if len(all_images) <= target_count:
        return source_dir, None
    
    # Otherwise, create a random sample
    sampled_images = random.sample(all_images, target_count)
    
    # Copy sampled images to temp directory
    print(f"Creating balanced sample of {target_count} images...")
    for i, img_path in enumerate(tqdm(sampled_images)):
        ext = Path(img_path).suffix
        shutil.copy(img_path, os.path.join(temp_dir, f"img_{i:06d}{ext}"))
    
    return temp_dir, temp_dir


def main():
    parser = argparse.ArgumentParser(description="Calculate KID between real and generated images")
    parser.add_argument("--real-path", type=str, required=True, help="Path to real images")
    parser.add_argument("--gen-path", type=str, required=True, help="Path to generated images")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for feature extraction")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of workers for data loading")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to use (cuda or cpu)")
    parser.add_argument("--no-balance", action="store_true", 
                        help="Don't balance the number of real and generated images")
    parser.add_argument("--kid-subsets", type=int, default=100,
                        help="Number of subsets to use for KID calculation")
    parser.add_argument("--kid-subset-size", type=int, default=1000,
                        help="Size of each subset for KID calculation")
    
    args = parser.parse_args()
    
    # Check if directories exist and contain images
    real_count = check_directory(args.real_path)
    gen_count = check_directory(args.gen_path)
    
    print(f"Found {real_count} real images and {gen_count} generated images")
    
    # Create balanced sample if needed
    temp_dir = None
    real_path_for_kid = args.real_path
    
    if not args.no_balance and real_count > gen_count:
        real_path_for_kid, temp_dir = create_balanced_sample(args.real_path, gen_count)
        if temp_dir:
            print(f"Created balanced sample with {gen_count} real images")
    
    # Determine the actual number of images that will be used
    actual_real_count = count_images(real_path_for_kid)
    actual_gen_count = gen_count
    min_count = min(actual_real_count, actual_gen_count)
    
    # Adjust KID subset size if necessary
    kid_subset_size = min(args.kid_subset_size, min_count)
    if kid_subset_size < args.kid_subset_size:
        print(f"Warning: Reducing KID subset size from {args.kid_subset_size} to {kid_subset_size} due to limited number of images")
    
    # Adjust KID subsets if necessary
    kid_subsets = min(args.kid_subsets, min_count // 2)  # Ensure we have enough samples for the requested subsets
    if kid_subsets < args.kid_subsets:
        print(f"Warning: Reducing KID subsets from {args.kid_subsets} to {kid_subsets} due to limited number of images")
    
    try:
        # Calculate KID score using torch-fidelity
        print("\nCalculating KID score...")
        
        # Extract model ID from the generated path if possible
        gen_path_parts = Path(args.gen_path).parts
        model_id = None
        for part in gen_path_parts:
            # Look for patterns like i2v_r128_250 in the path
            if '_' in part and any(char.isdigit() for char in part):
                model_id = part
                break
                
        if model_id:
            print(f"Model ID: {model_id}")
            
        # Calculate metrics
        calculate_metrics(
            input1=args.gen_path,
            input2=real_path_for_kid,
            cuda=args.device == "cuda",
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            kid=True,  # Only calculate KID
            kid_subsets=kid_subsets,
            kid_subset_size=kid_subset_size,
            verbose=True
        )
        
        # The KID value is printed by the calculate_metrics function
        print("\nKID calculation complete. See the value printed above.")
        
    finally:
        # Clean up temporary directory if created
        if temp_dir:
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary directory")


if __name__ == "__main__":
    main() 
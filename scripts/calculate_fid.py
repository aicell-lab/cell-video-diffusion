#!/usr/bin/env python3
"""
Calculate FID (Fréchet Inception Distance) scores between real and generated images.

This script computes the FID score between two directories of images, typically
containing real and generated images. By default, it will randomly sample from the
real images to match the number of generated images for a balanced comparison.

Usage:
    python calculate_fid.py --real_path <real_images_dir> --gen_path <generated_images_dir> [options]

Examples:
    # Calculate FID between real images and generated images
    python calculate_fid.py --real_path ../data/processed/idr0013/final_frames/ --gen_path ../data/generated/test_generations_plateval/i2v_r128_250/frame_final/

    # Calculate FID using GPU with a larger batch size
    python calculate_fid.py --real_path ../data/real_frames/ --gen_path ../data/generated_frames/ --device cuda --batch_size 100
    
    # Calculate FID without balancing the number of images
    python calculate_fid.py --real_path ../data/real_frames/ --gen_path ../data/generated_frames/ --no_balance
"""

import os
import argparse
import torch
import random
import shutil
import tempfile
from pathlib import Path
from pytorch_fid.fid_score import calculate_fid_given_paths


def count_images(directory):
    """
    Count the number of image files in a directory.
    
    Args:
        directory: Path to the directory
        
    Returns:
        int: Number of image files
    """
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']
    image_files = [f for f in Path(directory).iterdir() 
                  if f.is_file() and f.suffix.lower() in image_extensions]
    return len(image_files)


def get_image_files(directory):
    """
    Get a list of image files in a directory.
    
    Args:
        directory: Path to the directory
        
    Returns:
        list: List of image file paths
    """
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp']
    image_files = [f for f in Path(directory).iterdir() 
                  if f.is_file() and f.suffix.lower() in image_extensions]
    return image_files


def check_directory(path):
    """
    Check if a directory exists and contains image files.
    
    Args:
        path: Path to the directory
        
    Returns:
        bool: True if the directory exists and contains images, False otherwise
    """
    dir_path = Path(path)
    
    if not dir_path.exists():
        print(f"Error: Directory does not exist: {path}")
        return False
    
    if not dir_path.is_dir():
        print(f"Error: Path is not a directory: {path}")
        return False
    
    # Check for image files
    image_files = get_image_files(dir_path)
    
    if not image_files:
        print(f"Warning: No image files found in directory: {path}")
        return False
    
    return True


def create_balanced_sample(real_path, gen_path):
    """
    Create a temporary directory with a random sample of real images
    that matches the number of generated images.
    
    Args:
        real_path: Path to the directory containing real images
        gen_path: Path to the directory containing generated images
        
    Returns:
        str: Path to the temporary directory with sampled real images
    """
    # Count images
    num_gen_images = count_images(gen_path)
    real_images = get_image_files(real_path)
    num_real_images = len(real_images)
    
    if num_real_images <= num_gen_images:
        print(f"No sampling needed: {num_real_images} real images <= {num_gen_images} generated images")
        return real_path
    
    print(f"Sampling {num_gen_images} images from {num_real_images} real images...")
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix="fid_real_sample_")
    
    # Randomly sample real images
    sampled_images = random.sample(real_images, num_gen_images)
    
    # Copy sampled images to temporary directory
    for img_path in sampled_images:
        shutil.copy2(img_path, Path(temp_dir) / img_path.name)
    
    return temp_dir


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Calculate FID scores between real and generated images')
    parser.add_argument('--real_path', type=str, required=True,
                        help='Path to the directory containing real images')
    parser.add_argument('--gen_path', type=str, required=True,
                        help='Path to the directory containing generated images')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Batch size for feature extraction (default: 50)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use for computation: "cpu" or "cuda" (default: cpu)')
    parser.add_argument('--dims', type=int, default=2048,
                        help='Dimensionality of features (default: 2048 for Inception v3)')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of worker processes for data loading (default: 8)')
    parser.add_argument('--no_balance', action='store_true',
                        help='Do not balance the number of real and generated images')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for sampling (default: 42)')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Validate directories
    if not check_directory(args.real_path) or not check_directory(args.gen_path):
        return
    
    # Validate device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Falling back to CPU.")
        args.device = 'cpu'
    
    # Create a balanced sample if needed
    temp_dir = None
    real_path_for_fid = args.real_path
    
    if not args.no_balance:
        temp_dir = create_balanced_sample(args.real_path, args.gen_path)
        real_path_for_fid = temp_dir
    
    print(f"Calculating FID score...")
    print(f"Real images: {args.real_path}" + 
          (f" (sampled to {count_images(real_path_for_fid)} images)" if temp_dir else ""))
    print(f"Generated images: {args.gen_path} ({count_images(args.gen_path)} images)")
    print(f"Using device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    
    try:
        # Calculate FID score
        fid_value = calculate_fid_given_paths(
            paths=[real_path_for_fid, args.gen_path],
            batch_size=args.batch_size,
            device=args.device,
            dims=args.dims,
            num_workers=args.num_workers
        )
        
        print(f"\nFID score: {fid_value:.4f}")
        
        # Optionally save the result to a file
        result_dir = Path("fid_results")
        result_dir.mkdir(exist_ok=True)
        
        # Extract model ID from the generated path if possible
        gen_path_parts = Path(args.gen_path).parts
        model_id = None
        for part in gen_path_parts:
            # Look for patterns like i2v_r128_250 in the path
            if '_' in part and any(char.isdigit() for char in part):
                model_id = part
                break
        
        # Create a filename based on the directories and model ID
        real_name = Path(args.real_path).name
        gen_name = Path(args.gen_path).name
        balanced_suffix = "" if args.no_balance else "_balanced"
        
        # Include model ID in the filename if found
        if model_id:
            result_file = result_dir / f"fid_{model_id}_{real_name}_vs_{gen_name}{balanced_suffix}.txt"
        else:
            result_file = result_dir / f"fid_{real_name}_vs_{gen_name}{balanced_suffix}.txt"
        
        with open(result_file, 'w') as f:
            f.write(f"Model ID: {model_id if model_id else 'Unknown'}\n")
            f.write(f"Real images: {args.real_path}\n")
            if temp_dir:
                f.write(f"Real images sampled: {count_images(real_path_for_fid)}\n")
            f.write(f"Generated images: {args.gen_path}\n")
            f.write(f"Generated images count: {count_images(args.gen_path)}\n")
            f.write(f"FID score: {fid_value:.4f}\n")
        
        print(f"Results saved to {result_file}")
        
    except Exception as e:
        print(f"Error calculating FID score: {e}")
    
    finally:
        # Clean up temporary directory if created
        if temp_dir and temp_dir != args.real_path:
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary directory: {temp_dir}")


if __name__ == "__main__":
    main() 
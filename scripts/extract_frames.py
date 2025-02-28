#!/usr/bin/env python3
"""
Extract frames from video files.

This script extracts specific frames from video files in a directory structure
and saves them as PNG images in a specified output directory structure.

Usage:
    python extract_frames.py --input_dir <input_directory> --output_subdir <output_subdirectory> [options]

Examples:
    # Extract final frame from all videos in test_generations_plateval to a 'frame_final' subdirectory
    python extract_frames.py --input_dir ../data/generated/test_generations_plateval --output_subdir frame_final

    # Extract the 10th frame from all videos (1-based indexing)
    python extract_frames.py --input_dir ../data/generated/test_generations_plateval --output_subdir frame_10 --frame_position 10

    # Extract the first frame from all videos (using 1-based indexing)
    python extract_frames.py --input_dir ../data/generated/test_generations_plateval --output_subdir frame_first --frame_position 1
"""

import os
import re
import cv2
import argparse
from pathlib import Path


def extract_frame(video_path, output_path, frame_position='last'):
    """
    Extract a specific frame from a video and save it as a PNG using OpenCV.
    
    Args:
        video_path: Path to the video file
        output_path: Path where the frame will be saved
        frame_position: Which frame to extract. Can be:
                        - 'last': the last frame (default)
                        - int: a specific frame number (1-based indexing)
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Open the video file
    cap = cv2.VideoCapture(str(video_path))
    
    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return False
    
    # Get total frame count
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Determine which frame to extract
    if frame_position == 'last':
        frame_idx = total_frames - 1
    elif isinstance(frame_position, int):
        # Convert from 1-based to 0-based indexing
        frame_idx = frame_position - 1
        
        # Check if the frame index is valid
        if frame_idx < 0 or frame_idx >= total_frames:
            print(f"Error: Frame index {frame_position} out of range (1-{total_frames}) for {video_path}")
            cap.release()
            return False
    else:
        print(f"Error: Invalid frame position: {frame_position}")
        cap.release()
        return False
    
    # Set position to the desired frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    
    # Read the frame
    ret, frame = cap.read()
    
    # Check if frame was read successfully
    if not ret:
        print(f"Error: Could not read frame {frame_idx+1} from {video_path}")
        cap.release()
        return False
    
    # Save the frame as PNG
    cv2.imwrite(str(output_path), frame)
    
    # Release the video capture object
    cap.release()
    
    return True


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract frames from videos')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Base directory containing videos or subdirectories with videos')
    parser.add_argument('--output_subdir', type=str, required=True,
                        help='Name of subdirectory to create for output frames')
    parser.add_argument('--frame_position', type=str, default='last',
                        help='Which frame to extract: "last" or a specific frame number (1-based indexing)')
    parser.add_argument('--pattern', type=str, default=r'(LT\d+_\d+-\d+_\d+)_seed\d+_S\d+_G\d+_F\d+_FPS\d+\.mp4',
                        help='Regex pattern to extract base name from video filenames')
    parser.add_argument('--suffix', type=str, default='frame_final',
                        help='Suffix to add to the output filename')
    parser.add_argument('--no_recursive', action='store_true',
                        help='Do not process subdirectories recursively (default is to process recursively)')
    
    args = parser.parse_args()
    
    # Convert frame_position to int if it's a number
    if args.frame_position != 'last':
        try:
            args.frame_position = int(args.frame_position)
            if args.frame_position < 1:
                print(f"Error: frame_position must be 'last' or a positive number (1-based indexing), got {args.frame_position}")
                return
        except ValueError:
            print(f"Error: frame_position must be 'last' or a positive number, got {args.frame_position}")
            return
    
    # Base directory containing all the video directories
    base_dir = Path(args.input_dir)
    
    # Pattern to match video files
    video_pattern = re.compile(args.pattern)
    
    # Process directories - by default, process all subdirectories
    dirs_to_process = [base_dir]
    if not args.no_recursive:
        # Get all immediate subdirectories (not recursive)
        subdirs = [p for p in base_dir.iterdir() if p.is_dir()]
        if subdirs:  # If subdirectories found, use them instead of base directory
            dirs_to_process = subdirs
    
    print(f"Processing {len(dirs_to_process)} directories...")
    
    # Track statistics
    total_videos = 0
    total_frames_extracted = 0
    
    for dir_path in dirs_to_process:
        # Create output subdirectory
        frame_dir = dir_path / args.output_subdir
        os.makedirs(frame_dir, exist_ok=True)
        
        # Process each video file in the directory
        video_files = list(dir_path.glob("*.mp4"))
        if video_files:
            print(f"Processing {len(video_files)} videos in {dir_path}")
            
            for video_file in video_files:
                total_videos += 1
                match = video_pattern.match(video_file.name)
                if match:
                    # Extract the base name
                    base_name = match.group(1)
                    
                    # Create output path
                    output_path = frame_dir / f"{base_name}_{args.suffix}.png"
                    
                    # Extract the frame
                    print(f"  Extracting frame from {video_file.name}")
                    success = extract_frame(video_file, output_path, args.frame_position)
                    if success:
                        total_frames_extracted += 1
                else:
                    print(f"  Warning: Filename {video_file.name} doesn't match the expected pattern")
        else:
            print(f"No video files found in {dir_path}")
    
    print(f"\nSummary: Processed {total_videos} videos, extracted {total_frames_extracted} frames")


if __name__ == "__main__":
    main() 
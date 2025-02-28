#!/usr/bin/env python3
"""
Create combined frame images from video files.

This script extracts multiple frames from video files and combines them into grid layouts.
If start frame is 0, it will include the conditioning frame from the hardcoded directory.
It also adds the corresponding prompt at the top of each combined image.

Usage:
    python create_combined_frames.py --input_dir <input_directory> --start <start_frame> --end <end_frame> [options]

Examples:
    # Create combined images with frames 1 to 10 from all videos
    python create_combined_frames.py --input_dir ../data/generated/test_generations_plateval --start 1 --end 10

    # Create combined images with frames 0 to 10 (including conditioning frame)
    python create_combined_frames.py --input_dir ../data/generated/test_generations_plateval --start 0 --end 10
    python create_combined_frames.py --input_dir ../data/generated/test_generations_realval --start 0 --end 10

    # Create combined images with frames 0 to 16 with 4 frames per row
    python create_combined_frames.py --input_dir ../data/generated/test_generations_plateval --start 0 --end 16 --cols 4
"""

import os
import re
import cv2
import csv
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Hardcoded path to conditioning frames
COND_FRAMES_DIR = "/proj/aicell/users/x_aleho/video-diffusion/data/processed/idr0013/first_frames"

# Hardcoded path to prompt mapping CSV
PROMPT_MAPPING_CSV = "./idr0013/prompt_mapping_all.csv"

# Background color (light gray)
BACKGROUND_COLOR = '#f0f0f0'  # Light gray


def load_prompt_mapping():
    """
    Load the prompt mapping from CSV file.
    
    Returns:
        dict: Mapping from plate_well_id to prompt
    """
    prompt_mapping = {}
    
    try:
        with open(PROMPT_MAPPING_CSV, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                plate_well_id = row['plate_well_id']
                prompt = row['prompt']
                prompt_mapping[plate_well_id] = prompt
        
        print(f"Loaded {len(prompt_mapping)} prompts from {PROMPT_MAPPING_CSV}")
        
        if not prompt_mapping:
            print(f"Error: No prompts found in {PROMPT_MAPPING_CSV}")
            sys.exit(1)
            
        return prompt_mapping
    
    except Exception as e:
        print(f"Error loading prompt mapping: {e}")
        sys.exit(1)


def extract_frame(video_path, frame_position):
    """
    Extract a specific frame from a video.
    
    Args:
        video_path: Path to the video file
        frame_position: Which frame to extract (1-based indexing)
    
    Returns:
        numpy.ndarray: The extracted frame, or None if extraction failed
    """
    # Open the video file
    cap = cv2.VideoCapture(str(video_path))
    
    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
    
    # Get total frame count
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Convert from 1-based to 0-based indexing
    frame_idx = frame_position - 1
    
    # Check if the frame index is valid
    if frame_idx < 0 or frame_idx >= total_frames:
        print(f"Error: Frame index {frame_position} out of range (1-{total_frames}) for {video_path}")
        cap.release()
        return None
    
    # Set position to the desired frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    
    # Read the frame
    ret, frame = cap.read()
    
    # Release the video capture object
    cap.release()
    
    # Check if frame was read successfully
    if not ret:
        print(f"Error: Could not read frame {frame_position} from {video_path}")
        return None
    
    # Convert BGR to RGB (OpenCV uses BGR, matplotlib uses RGB)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    return frame


def load_conditioning_frame(base_name):
    """
    Load the conditioning frame (frame 0) for a given video.
    
    Args:
        base_name: Base name of the video (e.g., LT0001_12-00326_01)
    
    Returns:
        numpy.ndarray: The conditioning frame, or None if not found
    """
    # Construct the path to the conditioning frame
    cond_frame_path = Path(COND_FRAMES_DIR) / f"{base_name}.png"
    
    # Check if the file exists
    if not cond_frame_path.exists():
        print(f"Warning: Conditioning frame not found: {cond_frame_path}")
        return None
    
    # Load the image
    try:
        frame = cv2.imread(str(cond_frame_path))
        if frame is None:
            print(f"Error: Could not read conditioning frame: {cond_frame_path}")
            return None
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
    
    except Exception as e:
        print(f"Error loading conditioning frame {cond_frame_path}: {e}")
        return None


def create_combined_image(video_path, start_frame, end_frame, cols=5, base_name=None, prompt=None):
    """
    Create a combined image with multiple frames from a video.
    
    Args:
        video_path: Path to the video file
        start_frame: First frame to include (0-based indexing, where 0 is conditioning frame)
        end_frame: Last frame to include (1-based indexing for video frames)
        cols: Number of columns in the grid
        base_name: Base name of the video for finding conditioning frame
        prompt: Prompt text to display at the top of the image
    
    Returns:
        numpy.ndarray: The combined image, or None if creation failed
    """
    frames = []
    
    # Load conditioning frame (frame 0) if requested
    if start_frame == 0 and base_name:
        cond_frame = load_conditioning_frame(base_name)
        if cond_frame is not None:
            frames.append(cond_frame)
        else:
            print(f"Warning: Could not load conditioning frame for {base_name}")
    
    # Extract all requested frames from the video
    video_start = max(1, start_frame)  # Start from frame 1 if start_frame is 0
    for frame_num in range(video_start, end_frame + 1):
        frame = extract_frame(video_path, frame_num)
        if frame is not None:
            frames.append(frame)
        else:
            print(f"Warning: Skipping frame {frame_num} for {video_path}")
    
    if not frames:
        print(f"Error: No frames could be extracted from {video_path}")
        return None
    
    # Calculate grid dimensions
    num_frames = len(frames)
    rows = (num_frames + cols - 1) // cols  # Ceiling division
    
    # Create figure with extra space at the top for the prompt
    fig_height = rows * 3 + 0.5  # Always add extra space for the prompt
    
    # Create figure with light gray background
    fig = Figure(figsize=(cols * 3, fig_height), facecolor=BACKGROUND_COLOR)
    canvas = FigureCanvas(fig)
    
    # Add prompt at the top
    fig.suptitle(prompt, fontsize=12, wrap=True)
    
    # Create grid for frames
    grid = fig.add_gridspec(rows, cols)
    
    # Add frame number labels and frames to the grid
    for i, frame in enumerate(frames):
        row = i // cols
        col = i % cols
        
        # Add subplot with light gray background
        ax = fig.add_subplot(grid[row, col], facecolor=BACKGROUND_COLOR)
        
        # Display the frame
        ax.imshow(frame)
        
        # Add frame number as title
        if start_frame == 0 and i == 0:
            ax.set_title("Frame 0 (Conditioning)")
        else:
            # Adjust frame number if conditioning frame is included
            frame_num = i if start_frame > 0 else i + start_frame
            ax.set_title(f"Frame {frame_num}")
        
        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Set light gray edge color for the frame
        for spine in ax.spines.values():
            spine.set_edgecolor(BACKGROUND_COLOR)
    
    # Adjust layout
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    
    # Convert figure to image
    canvas.draw()
    combined_image = np.array(canvas.renderer.buffer_rgba())
    
    return combined_image


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Create combined frame images from videos')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Base directory containing videos or subdirectories with videos')
    parser.add_argument('--start', type=int, required=True,
                        help='First frame to include (0 for conditioning frame, 1+ for video frames)')
    parser.add_argument('--end', type=int, required=True,
                        help='Last frame to include (1-based indexing)')
    parser.add_argument('--cols', type=int, default=5,
                        help='Number of columns in the grid (default: 5)')
    parser.add_argument('--pattern', type=str, default=r'(LT\d+_\d+-\d+_\d+)_seed\d+_S\d+_G\d+_F\d+_FPS\d+\.mp4',
                        help='Regex pattern to extract base name from video filenames')
    parser.add_argument('--no_recursive', action='store_true',
                        help='Do not process subdirectories recursively (default is to process recursively)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.start < 0:
        print(f"Error: start frame must be 0 or a positive number, got {args.start}")
        return
    
    if args.end < args.start and args.start > 0:
        print(f"Error: end frame ({args.end}) must be greater than or equal to start frame ({args.start})")
        return
    
    if args.end < 1:
        print(f"Error: end frame must be a positive number, got {args.end}")
        return
    
    if args.cols < 1:
        print(f"Error: cols must be a positive number, got {args.cols}")
        return
    
    # Check if conditioning frames directory exists when start frame is 0
    if args.start == 0 and not Path(COND_FRAMES_DIR).is_dir():
        print(f"Error: Conditioning frames directory does not exist: {COND_FRAMES_DIR}")
        sys.exit(1)
    
    # Check if prompt mapping file exists
    if not Path(PROMPT_MAPPING_CSV).exists():
        print(f"Error: Prompt mapping file not found: {PROMPT_MAPPING_CSV}")
        sys.exit(1)
    
    # Load prompt mapping
    prompt_mapping = load_prompt_mapping()
    
    # Base directory containing all the video directories
    base_dir = Path(args.input_dir)
    
    # Pattern to match video files
    video_pattern = re.compile(args.pattern)
    
    # Create output subdirectory name
    output_subdir = f"frame_{args.start}_to_{args.end}"
    
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
    total_images_created = 0
    
    for dir_path in dirs_to_process:
        # Create output subdirectory
        frame_dir = dir_path / output_subdir
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
                    
                    # Get prompt - exit if not found
                    if base_name not in prompt_mapping:
                        print(f"Error: No prompt found for {base_name}")
                        sys.exit(1)
                    
                    prompt = prompt_mapping[base_name]
                    
                    # Create output path
                    output_path = frame_dir / f"{base_name}_frames_{args.start}_to_{args.end}.png"
                    
                    # Create combined image
                    print(f"  Creating combined image from {video_file.name}")
                    combined_image = create_combined_image(
                        video_file, args.start, args.end, args.cols, base_name, prompt
                    )
                    
                    if combined_image is not None:
                        # Save the combined image
                        plt.imsave(output_path, combined_image)
                        total_images_created += 1
                        print(f"  Saved combined image to {output_path}")
                    else:
                        print(f"  Failed to create combined image for {video_file.name}")
                else:
                    print(f"  Warning: Filename {video_file.name} doesn't match the expected pattern")
        else:
            print(f"No video files found in {dir_path}")
    
    print(f"\nSummary: Processed {total_videos} videos, created {total_images_created} combined images")


if __name__ == "__main__":
    main()
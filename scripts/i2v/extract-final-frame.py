#!/usr/bin/env python
# Script to extract the final frame from a video file or directory of videos
"""
Usage example:
  # For a single video:
  python extract-final-frame.py ../../data/processed/idr0013/LT0001_02/00223_01.mp4

  # For a directory of videos:
  python extract-final-frame.py ../../data/processed/idr0013/LT0001_02/ --dir

  # For generated videos:
  python extract-final-frame.py ../../data/processed/idr0013/LT0001_02/00223_01.mp4 --generated

  # For a directory of generated videos:
  python extract-final-frame.py /path/to/generated/videos/ --dir --generated
"""
import cv2
import argparse
import os
import sys
import glob

def extract_final_frame(video_path, output_path=None, generated=False):
    """
    Extract the final frame from a video file and save it as an image.
    
    Args:
        video_path (str): Path to the video file
        output_path (str, optional): Path to save the output image. If None, 
                                     will generate a path automatically
        generated (bool): Whether the video is generated or not
    
    Returns:
        bool: True if successful, False otherwise
    """
    # Check if the video file exists
    if not os.path.isfile(video_path):
        print(f"Error: Video file '{video_path}' does not exist.")
        return False
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video file '{video_path}'.")
        return False
    
    # Get total number of frames
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        print(f"Error: Could not determine frame count for '{video_path}'.")
        return False
    
    # Set the position to the last frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
    
    # Read the last frame
    ret, frame = cap.read()
    
    # Release the video capture object
    cap.release()
    
    if not ret:
        print(f"Error: Could not read the final frame from '{video_path}'.")
        return False
    
    # Resize the frame to 768x1360 if it's a real (non-generated) video
    if not generated:
        frame = cv2.resize(frame, (1360, 768), interpolation=cv2.INTER_AREA)
        print(f"Resized frame to 768x1360")
    
    # Determine output path if not provided
    if output_path is None:
        # Extract the base filename without extension
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # Create output directory structure
        if generated:
            # For generated videos, use a 'generated' subdirectory and keep the full filename
            output_dir = os.path.join(os.path.dirname(os.path.dirname(video_path)), 'final_frames', 'generated')
            output_filename = f"{base_name}_finalframe.png"
        else:
            # For regular videos, use the original structure with plate_id
            parent_dir = os.path.dirname(os.path.abspath(video_path))
            parent_dir_name = os.path.basename(parent_dir)
            output_dir = os.path.join(os.path.dirname(parent_dir), 'final_frames')
            output_filename = f"{parent_dir_name}_{base_name}.png"
        
        # Create the output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Set the output path
        output_path = os.path.join(output_dir, output_filename)
    
    # Save the frame
    try:
        cv2.imwrite(output_path, frame)
        print(f"Final frame saved to: {output_path}")
        return True
    except Exception as e:
        print(f"Error saving frame: {e}")
        return False

def process_directory(directory_path, generated=False):
    """
    Process all video files in a directory.
    
    Args:
        directory_path (str): Path to the directory containing video files
        generated (bool): Whether the videos are generated or not
    
    Returns:
        tuple: (success_count, total_count)
    """
    if not os.path.isdir(directory_path):
        print(f"Error: Directory '{directory_path}' does not exist.")
        return 0, 0
    
    # Get all video files in the directory
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []
    
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(directory_path, f'*{ext}')))
    
    if not video_files:
        print(f"No video files found in '{directory_path}'.")
        return 0, 0
    
    print(f"Found {len(video_files)} video files in '{directory_path}'.")
    
    # Process each video file
    success_count = 0
    for video_path in video_files:
        print(f"Processing: {video_path}")
        if extract_final_frame(video_path, generated=generated):
            success_count += 1
    
    return success_count, len(video_files)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Extract the final frame from a video file or directory of videos.')
    parser.add_argument('path', help='Path to the video file or directory')
    parser.add_argument('-o', '--output', help='Path to save the output image (only for single video)')
    parser.add_argument('-g', '--generated', action='store_true', 
                        help='Specify if the video(s) are generated (saves to final_frames/generated/)')
    parser.add_argument('-d', '--dir', action='store_true',
                        help='Process all video files in the directory')
    
    args = parser.parse_args()
    
    # Check if path exists
    if not os.path.exists(args.path):
        print(f"Error: Path '{args.path}' does not exist.")
        sys.exit(1)
    
    # Process directory or single file
    if args.dir:
        if not os.path.isdir(args.path):
            print(f"Error: '{args.path}' is not a directory.")
            sys.exit(1)
        
        if args.output:
            print("Warning: Output path is ignored when processing a directory.")
        
        success_count, total_count = process_directory(args.path, args.generated)
        print(f"Processed {total_count} videos, {success_count} successful.")
        
        # Exit with success if at least one video was processed successfully
        sys.exit(0 if success_count > 0 else 1)
    else:
        # Process single file
        if not os.path.isfile(args.path):
            print(f"Error: '{args.path}' is not a file.")
            sys.exit(1)
        
        success = extract_final_frame(args.path, args.output, args.generated)
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()

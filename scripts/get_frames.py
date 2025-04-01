#!/usr/bin/env python3

"""
get_frames.py

A utility script used specifically for extracting frames from cell videos for paper figures.
Extracts 5 frames (0, 20, 40, 60, 80) from each video and optionally enhances them with
colormap application to improve visibility of cellular structures.

Usage:
  python get_frames.py [--apply_color] [--colormap COLORMAP] [--enhance_contrast]
  
Arguments:
  --apply_color: Apply a colormap to enhance visibility (default: False)
  --colormap: OpenCV colormap to apply (default: COLORMAP_HOT)
  --enhance_contrast: Enhance contrast before applying colormap (default: False)
  
Notes:
  - Used specifically for extracting images for paper figures and comparisons
  - Hardcoded to extract 5 evenly-spaced frames from each video
  - Can process multiple videos with different labels (baseline, sft, real, etc.)
  - Output files are named using the video label and frame number (e.g., "baseline0.jpg")
"""

import cv2
import os
import argparse

def extract_frames(video_path, output_dir, video_label=None, apply_color=True, colormap=cv2.COLORMAP_JET, enhance_contrast=True):
    """
    Extract 5 frames from a video file: first frame, frame 20, 40, 60, and 80.
    Apply color enhancement to improve nuclei visibility.
    
    Args:
        video_path: Path to the MP4 video file
        output_dir: Directory to save the extracted frames
        video_label: Optional label to include in output filename (e.g., "baseline", "sft")
        apply_color: Whether to apply colormap to the frames
        colormap: OpenCV colormap to apply (default: COLORMAP_JET)
        enhance_contrast: Whether to enhance contrast before applying colormap
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Frames to extract
    frames_to_extract = [0, 20, 40, 60, 80]
    
    for frame_num in frames_to_extract:
        # Set the frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        
        # Read the frame
        ret, frame = cap.read()
        
        if ret:
            # Process the frame if color enhancement is requested
            if apply_color:
                # Check if frame is already grayscale, if not convert it
                if len(frame.shape) == 3:
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    gray_frame = frame.copy()
                
                # Enhance contrast if requested
                if enhance_contrast:
                    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
                    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                    gray_frame = clahe.apply(gray_frame)
                
                # Apply colormap
                colored_frame = cv2.applyColorMap(gray_frame, colormap)
                
                # Save the enhanced frame with simplified naming
                output_path = os.path.join(output_dir, f"{video_label}{frame_num}.jpg")
                cv2.imwrite(output_path, colored_frame)
                print(f"Saved frame {frame_num} to {output_path}")
            else:
                # Save original frame with simplified naming
                output_path = os.path.join(output_dir, f"{video_label}{frame_num}.jpg")
                cv2.imwrite(output_path, frame)
                print(f"Saved frame {frame_num} to {output_path}")
        else:
            print(f"Error: Could not read frame {frame_num}")
    
    # Release the video capture object
    cap.release()
    print("Frame extraction complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract 5 frames from a video file")
    parser.add_argument("--apply_color", action="store_true", help="Apply colormap to enhance visibility")
    parser.add_argument("--colormap", type=int, default=cv2.COLORMAP_HOT, 
                        help="OpenCV colormap to apply (default: COLORMAP_VIRIDIS)")
    parser.add_argument("--enhance_contrast", action="store_true", help="Enhance contrast before applying colormap")
    
    args = parser.parse_args()
    
    output_dir = "/proj/aicell/users/x_aleho/video-diffusion/data/generated/test_generations_i2v/for_paper/frames"

    # video_groups = [
    #     {"base": "/proj/aicell/users/x_aleho/video-diffusion/data/generated/test_generations_i2v/i2v_baseline/LT0001_02-00266_01_seed9_S50_G8_F81_FPS10.mp4",
    #     "sft": "/proj/aicell/users/x_aleho/video-diffusion/data/generated/test_generations_i2v/sft_i2v_900/LT0001_02-00266_01_seed9_S50_G8_F81_FPS10.mp4",
    #     "lora": "/proj/aicell/users/x_aleho/video-diffusion/data/generated/test_generations_i2v/i2v_r256_900/LT0001_02-00266_01_seed9_S50_G8_F81_FPS10.mp4",
    #     "real": "/proj/aicell/users/x_aleho/video-diffusion/data/processed/idr0013/LT0001_02/00266_01.mp4"}
    # ]
    video_groups = [
        {"baseline": "/proj/aicell/users/x_aleho/video-diffusion/data/generated/final_evals/base/seed1.mp4",
         "sft": "/proj/aicell/users/x_aleho/video-diffusion/data/generated/test_generations_i2v/sft_i2v_900/LT0001_02-00266_01_seed9_S50_G8_F81_FPS10.mp4",
         "real": "/proj/aicell/users/x_aleho/video-diffusion/data/processed/idr0013/LT0001_02/00266_01.mp4"
         }
    ]
    for video_group in video_groups:
        for video_label, video_path in video_group.items():
            extract_frames(video_path, output_dir, video_label=video_label, apply_color=args.apply_color, 
                          colormap=args.colormap, enhance_contrast=args.enhance_contrast)


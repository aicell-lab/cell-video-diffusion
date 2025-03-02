import cv2
import os
import argparse

def extract_frames(video_path, output_dir):
    """
    Extract 5 frames from a video file: first frame, frame 20, 40, 60, and 80.
    
    Args:
        video_path: Path to the MP4 video file
        output_dir: Directory to save the extracted frames
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return
    
    # Get video filename without extension
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Frames to extract
    frames_to_extract = [0, 20, 40, 60, 80]
    
    for frame_num in frames_to_extract:
        # Set the frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        
        # Read the frame
        ret, frame = cap.read()
        
        if ret:
            # Save the frame
            output_path = os.path.join(output_dir, f"{video_name}_frame_{frame_num}.jpg")
            cv2.imwrite(output_path, frame)
            print(f"Saved frame {frame_num} to {output_path}")
        else:
            print(f"Error: Could not read frame {frame_num}")
    
    # Release the video capture object
    cap.release()
    print("Frame extraction complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract 5 frames from a video file")
    parser.add_argument("video_path", help="Path to the MP4 video file")
    parser.add_argument("--output_dir", default="frames", help="Directory to save the extracted frames")
    
    args = parser.parse_args()
    extract_frames(args.video_path, args.output_dir)


"/proj/aicell/users/x_aleho/video-diffusion/CogVideo/models/loras/IDR0013-10plates-i2v-r128-a64/validation_res/validation-gen-500-0-<ALEXANDER>-Time-lapse-mi-23565.mp4"
"/proj/aicell/users/x_aleho/video-diffusion/data/generated/test_generations_realval/i2v_baseline/LT0001_02-00122_01_seed9_S50_G8_F81_FPS10.mp4"

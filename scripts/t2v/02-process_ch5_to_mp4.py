#!/usr/bin/env python3

"""
02-process_ch5_to_mp4.py

Processes CellH5 (.ch5) files identified in the filtered extreme phenotypes CSV
and converts them to MP4 video files. The script locates the image timeseries
in each file and creates a video showing cell behavior over time.

Usage:
  python 02-process_ch5_to_mp4.py /path/to/plates_dir output_dir [fps=10]
  
Arguments:
  plates_dir: Directory containing all the plate subdirectories with hdf5 folders
  output_dir: Directory where all MP4s will be placed (flat structure)
  fps: Frames per second for output videos (default: 10)
  
Notes:
  - Requires running 01-filter_extreme_phenotypes.py first to generate the input CSV
  - Creates a new CSV with video metadata at output/extreme_phenotypes_with_videos.csv
"""

import os
import sys
import glob
import h5py
import numpy as np
import imageio
import pandas as pd
import cv2

# Hardcoded path to the filtered extreme phenotypes file
FILTERED_SAMPLES_PATH = "/proj/aicell/users/x_aleho/video-diffusion/scripts/t2v/output/filtered_extreme_phenotypes.csv"

def find_ch5_image_datasets(h5file):
    """
    Search an open HDF5 file (CellH5) for datasets likely to contain raw images,
    typically (1, T, 1, Y, X) with dtype=uint8.

    Returns a list of dataset paths that match these criteria.
    """
    found_paths = []

    def visit_dataset(name, node):
        if isinstance(node, h5py.Dataset):
            shape = node.shape
            dtype = node.dtype
            # Check for 5D shape: (1, T, 1, Y, X) and dtype=uint8
            if (len(shape) == 5 
                and shape[0] == 1
                and shape[2] == 1
                and dtype == np.uint8):
                found_paths.append(name)

    h5file.visititems(visit_dataset)
    return found_paths

def load_ch5_as_timeseries(ch5_path):
    """
    Attempt to load the first matching image dataset from a MitoCheck CH5 file.
    Returns a 3D NumPy array of shape (T, Y, X) if found, else None.
    """
    with h5py.File(ch5_path, "r") as f:
        image_paths = find_ch5_image_datasets(f)
        if not image_paths:
            print(f"  No raw image dataset found in {ch5_path}")
            return None
        
        # We'll just take the first found path
        ds_path = image_paths[0]
        ds = f[ds_path]
        data_5d = ds[()]  # shape: (1, T, 1, Y, X)
        data_squeezed = np.squeeze(data_5d)  # (T, Y, X)
        return data_squeezed


def make_mp4(timeseries, out_path, fps=10, max_frames=None, target_size=None):
    """
    Write a 3D numpy array (T, Y, X) to an MP4 file.
    No per-frame normalization—use raw pixel values.
    
    Parameters:
    - timeseries: 3D numpy array (T, Y, X)
    - out_path: Path to save the MP4 file
    - fps: Frames per second
    - max_frames: Maximum number of frames to include (None = all frames)
    - target_size: Tuple (height, width) to resize frames (None = original size)
    """    
    print(f"Creating MP4 with data shape={timeseries.shape}, dtype={timeseries.dtype}")

    # Optional check: warn if not uint8
    if timeseries.dtype != np.uint8:
        print("WARNING: Data is not uint8. Consider scaling or casting to avoid unexpected clipping.")

    # Limit number of frames if specified
    if max_frames is not None and max_frames < timeseries.shape[0]:
        timeseries = timeseries[:max_frames]
        print(f"  Limited to first {max_frames} frames")
    
    T = timeseries.shape[0]
    writer = imageio.get_writer(out_path, fps=fps)
    
    # Import OpenCV only when needed
    if target_size is not None and timeseries.shape[1:] != target_size:
        need_resize = True
        # OpenCV uses (width, height) order, opposite of our (height, width)
        cv2_size = (target_size[1], target_size[0])
        print(f"  Resizing frames from {timeseries.shape[1:]} to {target_size}")
    else:
        need_resize = False
    
    for t in range(T):
        frame = timeseries[t]
        
        # Resize if needed using OpenCV (much faster than scikit-image)
        if need_resize:
            frame = cv2.resize(frame, cv2_size, interpolation=cv2.INTER_AREA)
        
        writer.append_data(frame)
    
    writer.close()

def format_well_number(number):
    """
    Convert a well number (like 20) to the format used in CH5 filenames (like "00020_01")
    """
    return f"{int(number):05d}_01"

def main():
    if len(sys.argv) < 3:
        print("Usage: python 02-process_ch5_to_mp4.py /path/to/plates_dir output_dir [fps=10]")
        print("  plates_dir: Directory containing all the plate subdirectories with hdf5 folders")
        print("  output_dir: Directory where all MP4s will be placed (flat structure)")
        print(f"  Using hardcoded samples file: {FILTERED_SAMPLES_PATH}")
        sys.exit(1)
    
    plates_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    fps = 10
    if len(sys.argv) >= 4:
        fps = int(sys.argv[3])
    
    # Create output directory if it doesn't exist
    output_dir = os.path.abspath(output_dir)  # Convert to absolute path
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    sample_set = set()
    sample_metadata = {}
    processed_info = []  # New list to collect processed video information
    
    # Load filtered samples with extreme phenotypes
    if os.path.exists(FILTERED_SAMPLES_PATH):
        samples_df = pd.read_csv(FILTERED_SAMPLES_PATH)
        
        # Print some sample values to help debug
        print("Sample well_number values from CSV:")
        print(samples_df['well_number'].head(5).tolist())
    else:
        print(f"Error: Filtered samples file not found at {FILTERED_SAMPLES_PATH}")
        print("Run the 01-filter_extreme_phenotypes.py script first.")
        sys.exit(1)
    
    # Create a set of (plate, formatted_well_number) tuples for quick lookup
    if 'well_number' in samples_df.columns:
        # Convert well_number to match CH5 filename format (e.g., 20 → "00020_01")
        samples_df['formatted_well_number'] = samples_df['well_number'].apply(format_well_number)
        
        # Create set of (plate, formatted_well_number) tuples
        sample_set = set(zip(samples_df['Plate'], samples_df['formatted_well_number']))
        
        # Also create a dictionary for metadata lookup
        sample_metadata = {}
        for _, row in samples_df.iterrows():
            key = (row['Plate'], row['formatted_well_number'])
            metadata = {
                'cell_death': row['cell_death_label'],
                'migration_speed': row['migration_speed_label'],
                'proliferation': row['proliferation_label'],
                'extreme_count': row['extreme_count'],
                'well': row['Well'],  # Original well ID (like A5)
                'well_number': row['well_number'],  # Original numeric format
                'row_idx': _  # Store the row index for later mapping
            }
            sample_metadata[key] = metadata
    else:
        print("Error: 'well_number' column not found in samples file.")
        print("Run the updated 01-filter_extreme_phenotypes.py script first.")
        sys.exit(1)
    
    print(f"Loaded {len(sample_set)} extreme phenotype samples to process")
    print(f"Example formatted well numbers: {list(samples_df['formatted_well_number'].head(3))}")
    
    # Set parameters for video processing
    max_frames = 81  # Limit to first 81 frames
    target_size = (768, 1360)  # Height, Width
    
    # Find all plate directories
    plate_dirs = sorted(glob.glob(os.path.join(plates_dir, "LT*--*")))
    
    if not plate_dirs:
        print(f"No plate directories found in {plates_dir}")
        sys.exit(1)
    
    processed_count = 0
    skipped_count = 0
    
    # Process each plate directory
    for plate_dir in plate_dirs:
        plate_name = os.path.basename(plate_dir).split('--')[0]  # Extract LT0001_02 part
        hdf5_dir = os.path.join(plate_dir, "hdf5")
        
        if not os.path.exists(hdf5_dir):
            print(f"No hdf5 directory found in {plate_dir}")
            continue
        
        # Check if we have any samples for this plate
        plate_samples = [s for s in sample_set if s[0] == plate_name]
        if not plate_samples:
            print(f"No extreme samples to process in plate {plate_name}, skipping")
            continue
        
        print(f"Processing plate {plate_name} - found {len(plate_samples)} extreme samples to process")
        
        # Find all .ch5 files in the hdf5 directory
        ch5_files = sorted(glob.glob(os.path.join(hdf5_dir, "*.ch5")))
        
        if not ch5_files:
            print(f"No .ch5 files found in {hdf5_dir}")
            continue
        
        # Print a few filenames to help with debugging
        if len(ch5_files) > 0:
            print(f"CH5 file examples: {[os.path.basename(f) for f in ch5_files[:3]]}")
            
            # Also print some sample_set items for this plate to compare formats
            plate_sample_keys = [(p, w) for p, w in sample_set if p == plate_name]
            if plate_sample_keys:
                print(f"Sample keys for this plate: {plate_sample_keys[:3]}")
        
        for ch5_file in ch5_files:
            well_number = os.path.splitext(os.path.basename(ch5_file))[0]  # e.g. "00046_01"
            
            # Check if this file matches one of our extreme samples
            key = (plate_name, well_number)
            if key in sample_set:
                # Get metadata for this sample
                metadata = sample_metadata.get(key, {})
                
                # Create filename with phenotype info
                phenotypes = []
                if 'cell_death' in metadata:
                    phenotypes.append(f"cd-{metadata['cell_death']}")
                if 'migration_speed' in metadata:
                    phenotypes.append(f"ms-{metadata['migration_speed']}")
                if 'proliferation' in metadata:
                    phenotypes.append(f"pr-{metadata['proliferation']}")
                
                phenotype_str = "_".join(phenotypes)
                
                # Create output filename: plate-wellNum-phenotypes.mp4
                out_filename = f"{plate_name}-{well_number}-{phenotype_str}.mp4"
                out_mp4_path = os.path.join(output_dir, out_filename)
                
                # Check if MP4 already exists
                if os.path.exists(out_mp4_path):
                    print(f"MP4 already exists, skipping processing: {out_mp4_path}")
                    skipped_count += 1
                    
                    # Still add to processed_info even though we skipped processing
                    processed_info.append({
                        'plate': plate_name,
                        'well_number': metadata['well_number'],  # Original well number
                        'well': metadata['well'],  # Original well ID (like A5)
                        'cell_death_label': metadata['cell_death'],
                        'migration_speed_label': metadata['migration_speed'],
                        'proliferation_label': metadata['proliferation'],
                        'cell_death_score': samples_df.loc[metadata['row_idx'], 'cell_death'],
                        'migration_speed_score': samples_df.loc[metadata['row_idx'], 'migration_speed'],
                        'proliferation_score': samples_df.loc[metadata['row_idx'], 'proliferation'],
                        'extreme_count': metadata['extreme_count'],
                        'video_path': os.path.abspath(out_mp4_path),  # Absolute path
                        'video_filename': out_filename,  # Just the filename
                        'ch5_path': ch5_file  # Original ch5 file
                    })
                else:
                    print(f"Processing {ch5_file} -> {out_mp4_path}")
                    try:
                        data_squeezed = load_ch5_as_timeseries(ch5_file)
                        if data_squeezed is None:
                            print("  No data found, skipping")
                            continue
                        
                        print(f"  shape={data_squeezed.shape}, dtype={data_squeezed.dtype}")
                        
                        make_mp4(data_squeezed, out_mp4_path, fps=fps, max_frames=max_frames, target_size=target_size)
                        print(f"  Wrote {out_mp4_path}")
                        
                        # Store information about the processed video
                        processed_info.append({
                            'plate': plate_name,
                            'well_number': metadata['well_number'],  # Original well number
                            'well': metadata['well'],  # Original well ID (like A5)
                            'cell_death_label': metadata['cell_death'],
                            'migration_speed_label': metadata['migration_speed'],
                            'proliferation_label': metadata['proliferation'],
                            'cell_death_score': samples_df.loc[metadata['row_idx'], 'cell_death'],
                            'migration_speed_score': samples_df.loc[metadata['row_idx'], 'migration_speed'],
                            'proliferation_score': samples_df.loc[metadata['row_idx'], 'proliferation'],
                            'extreme_count': metadata['extreme_count'],
                            'video_path': os.path.abspath(out_mp4_path),  # Absolute path
                            'video_filename': out_filename,  # Just the filename
                            'ch5_path': ch5_file  # Original ch5 file
                        })
                        
                        processed_count += 1
                    except Exception as e:
                        print(f"  Error processing {ch5_file}: {str(e)}")
    
    print(f"Finished processing {processed_count} extreme phenotype samples")
    print(f"Skipped {skipped_count} already processed samples")
    print(f"Output directory: {output_dir}")
    
    # Save the extended CSV with video paths
    if processed_info:
        # Create a DataFrame with the processed information
        processed_df = pd.DataFrame(processed_info)
        
        # Save to a new CSV file in the output directory
        video_csv_path = os.path.join(os.path.dirname(FILTERED_SAMPLES_PATH), "extreme_phenotypes_with_videos.csv")
        processed_df.to_csv(video_csv_path, index=False)
        print(f"Saved video paths to: {video_csv_path}")
        print(f"Found and processed {len(processed_df)} videos")
    else:
        print("No videos were processed successfully")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3

"""
split_videos_file.py

A simple utility to split a videos.txt file into N chunks.

Usage:
  python split_videos_file.py --input_file path/to/videos.txt --output_dir chunks/ --num_chunks 8
"""

import os
import argparse
import math

def main():
    parser = argparse.ArgumentParser(description="Split a videos.txt file into N chunks.")
    parser.add_argument("--input_file", required=True, help="Path to videos.txt file")
    parser.add_argument("--output_dir", required=True, help="Directory to save chunks")
    parser.add_argument("--num_chunks", type=int, default=8, help="Number of chunks to create")
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Read all lines from input file
    with open(args.input_file, 'r') as f:
        all_lines = [line.strip() for line in f if line.strip()]
    
    total_lines = len(all_lines)
    print(f"Read {total_lines} video paths from {args.input_file}")
    
    # Calculate chunk size (round up)
    chunk_size = math.ceil(total_lines / args.num_chunks)
    
    # Split into chunks and write files
    for i in range(args.num_chunks):
        start = i * chunk_size
        end = min(start + chunk_size, total_lines)
        
        # Skip empty chunks
        if start >= total_lines:
            break
            
        chunk_lines = all_lines[start:end]
        basename = os.path.basename(args.input_file).split('.')[0]
        output_file = os.path.join(args.output_dir, f"{basename}_chunk{i+1}.txt")
        
        with open(output_file, 'w') as f:
            f.write('\n'.join(chunk_lines))
        
        print(f"Wrote chunk {i+1}/{args.num_chunks}: {len(chunk_lines)} lines to {output_file}")

if __name__ == "__main__":
    main() 
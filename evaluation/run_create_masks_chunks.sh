#!/bin/bash
#SBATCH -A berzelius-2025-23    # Your project/account
#SBATCH --gpus=1
#SBATCH -t 2-00:00:00            # Time limit (e.g. 1 day)
#SBATCH -J create_masks_chunks         # Job name
#SBATCH -o logs/%x_%j.out        # Standard output log
#SBATCH -e logs/%x_%j.err        # Standard error log

module load Mambaforge/23.3.1-1-hpc1-bdist

conda activate /proj/aicell/users/x_aleho/conda_envs/cogvideo

# run_create_masks.sh - SIMPLIFIED VERSION
# First run split_videos_file.py to create your chunks
# Then run this script which will process all chunks in parallel

SCRIPT_PATH="create_masks.py"       # Path to create_masks.py
OUTPUT_BASE_DIR="masks_output/t2v"      # Where mask files will be saved
LOG_DIR="logs/masks"                      # Where to store logs

mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_BASE_DIR}"

# MANUALLY LIST YOUR CHUNK FILES HERE
CHUNK_FILES=(
  "chunks/videos_chunk1.txt"
  "chunks/videos_chunk2.txt" 
  "chunks/videos_chunk3.txt"
  "chunks/videos_chunk4.txt"
  "chunks/videos_chunk5.txt"
  "chunks/videos_chunk6.txt"
  "chunks/videos_chunk7.txt"
  "chunks/videos_chunk8.txt"
  "chunks/videos_chunk9.txt"
  "chunks/videos_chunk10.txt"
  "chunks/videos_chunk11.txt"
  "chunks/videos_chunk12.txt"
  "chunks/videos_chunk13.txt"
  "chunks/videos_chunk14.txt"
  "chunks/videos_chunk15.txt"
)

# OUTPUT DIRECTORY
OUTPUT_DIR="${OUTPUT_BASE_DIR}/IDR0013-FILTERED-Test"
mkdir -p "${OUTPUT_DIR}"

# LAUNCH ONE JOB PER CHUNK
for CHUNK_FILE in "${CHUNK_FILES[@]}"; do
  if [[ ! -f "${CHUNK_FILE}" ]]; then
    echo "Warning: Chunk file not found: ${CHUNK_FILE}"
    continue
  fi
  
  CHUNK_NAME=$(basename "${CHUNK_FILE}" .txt)
  LOG_FILE="${LOG_DIR}/${CHUNK_NAME}.log"
  
  echo "Launching job for ${CHUNK_FILE}"
  python "${SCRIPT_PATH}" \
    --input_file "${CHUNK_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    > "${LOG_FILE}" 2>&1 &
done

# Wait for all background jobs to finish
wait

echo "All mask creation jobs have completed."
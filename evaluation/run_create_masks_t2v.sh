#!/bin/bash
#SBATCH -A berzelius-2025-23    # Your project/account
#SBATCH --gpus=1
#SBATCH -t 2-00:00:00            # Time limit (e.g. 1 day)
#SBATCH -J create_masks_t2v     # Job name
#SBATCH -o logs/%x_%j.out        # Standard output log
#SBATCH -e logs/%x_%j.err        # Standard error log

module load Mambaforge/23.3.1-1-hpc1-bdist

conda activate /proj/aicell/users/x_aleho/conda_envs/cogvideo

# Create necessary directories
SCRIPT_PATH="create_masks.py"
VIDEO_BASE_DIR="/proj/aicell/users/x_aleho/video-diffusion/data/generated/test_generations"
OUTPUT_BASE_DIR="masks_output/IDR0013-FILTERED"
LOG_DIR="logs"

mkdir -p "${LOG_DIR}"
mkdir -p "${OUTPUT_BASE_DIR}-pr-LOW"
mkdir -p "${OUTPUT_BASE_DIR}-pr-MED"
mkdir -p "${OUTPUT_BASE_DIR}-pr-HIGH"

# Create temporary file lists for each proliferation level
LOW_FILES=$(mktemp)
MED_FILES=$(mktemp)
HIGH_FILES=$(mktemp)

# Find all videos matching each proliferation level pattern
find "${VIDEO_BASE_DIR}" -name "t2v_pr-LOW_*.mp4" > "${LOW_FILES}"
find "${VIDEO_BASE_DIR}" -name "t2v_pr-MED_*.mp4" > "${MED_FILES}"
find "${VIDEO_BASE_DIR}" -name "t2v_pr-HIGH_*.mp4" > "${HIGH_FILES}"

# Report counts
LOW_COUNT=$(wc -l < "${LOW_FILES}")
MED_COUNT=$(wc -l < "${MED_FILES}")
HIGH_COUNT=$(wc -l < "${HIGH_FILES}")

echo "Found ${LOW_COUNT} LOW proliferation videos"
echo "Found ${MED_COUNT} MEDIUM proliferation videos"
echo "Found ${HIGH_COUNT} HIGH proliferation videos"

# Process each proliferation level
echo "Processing LOW proliferation videos..."
python "${SCRIPT_PATH}" \
  --input_file "${LOW_FILES}" \
  --output_dir "${OUTPUT_BASE_DIR}-pr-LOW" \
  > "${LOG_DIR}/mask_creation_LOW.log" 2>&1 &

echo "Processing MEDIUM proliferation videos..."
python "${SCRIPT_PATH}" \
  --input_file "${MED_FILES}" \
  --output_dir "${OUTPUT_BASE_DIR}-pr-MED" \
  > "${LOG_DIR}/mask_creation_MED.log" 2>&1 &

echo "Processing HIGH proliferation videos..."
python "${SCRIPT_PATH}" \
  --input_file "${HIGH_FILES}" \
  --output_dir "${OUTPUT_BASE_DIR}-pr-HIGH" \
  > "${LOG_DIR}/mask_creation_HIGH.log" 2>&1 &

# Wait for all background jobs to finish
wait

# Clean up temporary files
rm "${LOW_FILES}" "${MED_FILES}" "${HIGH_FILES}"

echo "All mask creation jobs have completed." 
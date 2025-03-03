#!/bin/bash
#SBATCH -A berzelius-2025-23    # Your project/account
#SBATCH --gpus=1 -C "thin"
#SBATCH -t 2-00:00:00            # Time limit (e.g. 1 day)
#SBATCH -J create_masks         # Job name
#SBATCH -o logs/%x_%j.out        # Standard output log
#SBATCH -e logs/%x_%j.err        # Standard error log

module load Mambaforge/23.3.1-1-hpc1-bdist

conda activate /proj/aicell/users/x_aleho/conda_envs/cogvideo

# run_create_masks.sh
#
# Example usage:
#   bash run_create_masks.sh
#
# This script:
# 1) Iterates through a list of directories, each containing .mp4 files.
# 2) Calls create_masks.py for each directory in parallel (background).
# 3) Saves logs for each run in a logs/ subdirectory.

SCRIPT_PATH="create_masks.py"    # Path to your create_masks.py
OUTPUT_BASE_DIR="masks_output"   # Where each directory's .npy files go
LOG_DIR="logs"                   # Where to store each run's log

# List of directories to process
DIRS=(
  # "/proj/aicell/users/x_aleho/video-diffusion/data/processed/idr0013/val/checkpoint-900-val"
  # "/proj/aicell/users/x_aleho/video-diffusion/data/generated/test_generations_realval/i2v_r64_150"
  # "/proj/aicell/users/x_aleho/video-diffusion/data/generated/test_generations_realval/i2v_r64_250"
  # "/proj/aicell/users/x_aleho/video-diffusion/data/generated/test_generations_realval/i2v_r64_500"
  # "/proj/aicell/users/x_aleho/video-diffusion/data/generated/test_generations_realval/i2v_r64_750"
  # "/proj/aicell/users/x_aleho/video-diffusion/data/generated/test_generations_realval/i2v_r64_900"
  # "/proj/aicell/users/x_aleho/video-diffusion/data/generated/test_generations_realval/i2v_r128_250"
  # "/proj/aicell/users/x_aleho/video-diffusion/data/generated/test_generations_realval/i2v_r128_500"
  # "/proj/aicell/users/x_aleho/video-diffusion/data/generated/test_generations_realval/i2v_r128_750"
  # "/proj/aicell/users/x_aleho/video-diffusion/data/generated/test_generations_realval/i2v_r128_900"
  # "/proj/aicell/users/x_aleho/video-diffusion/data/generated/test_generations_realval/i2v_r256_150"
  # "/proj/aicell/users/x_aleho/video-diffusion/data/generated/test_generations_realval/i2v_r256_250"
  # "/proj/aicell/users/x_aleho/video-diffusion/data/generated/test_generations_realval/i2v_r256_375"
  # "/proj/aicell/users/x_aleho/video-diffusion/data/generated/test_generations_realval/i2v_r256_750"
  # "/proj/aicell/users/x_aleho/video-diffusion/data/generated/test_generations_realval/i2v_r256_900"
  # "/proj/aicell/users/x_aleho/video-diffusion/data/generated/test_generations_realval/i2v_baseline"
  "/proj/aicell/users/x_aleho/video-diffusion/data/generated/test_generations_realval/sft_i2v_250"
  "/proj/aicell/users/x_aleho/video-diffusion/data/generated/test_generations_realval/sft_i2v_500"
  "/proj/aicell/users/x_aleho/video-diffusion/data/generated/test_generations_realval/sft_i2v_750"
  "/proj/aicell/users/x_aleho/video-diffusion/data/generated/test_generations_realval/sft_i2v_900"
)

mkdir -p "${LOG_DIR}"

#############################
# LAUNCH JOBS IN PARALLEL  #
#############################

for DIR in "${DIRS[@]}"; do
  BASENAME=$(basename "$DIR")  # e.g. 'i2v_r64_500'
  OUTPUT_DIR="${OUTPUT_BASE_DIR}/${BASENAME}"
  LOG_FILE="${LOG_DIR}/${BASENAME}.log"

  echo "Launching mask creation for: ${DIR}"
  python "${SCRIPT_PATH}" \
    --input_dir "${DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    > "${LOG_FILE}" 2>&1 &

done

# Wait for all background jobs to finish
wait

echo "All create_masks.py jobs have completed."
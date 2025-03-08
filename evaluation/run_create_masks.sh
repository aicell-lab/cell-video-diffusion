#!/bin/bash
#SBATCH -A berzelius-2025-23    # Your project/account
#SBATCH --gpus=1
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
OUTPUT_BASE_DIR="masks_output/t2v"   # Where each directory's .npy files go
LOG_DIR="logs"                   # Where to store each run's log

# List of directories to process - NO COMMAS between array items
DIRS=(
#  "/proj/aicell/users/x_aleho/video-diffusion/data/generated/final_evals/phenotype_alive/alive"
#  "/proj/aicell/users/x_aleho/video-diffusion/data/generated/final_evals/phenotype_dead/dead"
#  "/proj/aicell/users/x_aleho/video-diffusion/data/generated/final_evals/phenotype_cc/cc-HIGH"
#  "/proj/aicell/users/x_aleho/video-diffusion/data/generated/final_evals/phenotype_cc/cc-LOW"
#  "/proj/aicell/users/x_aleho/video-diffusion/data/generated/final_evals/phenotype_ms/ms-HIGH"
#  "/proj/aicell/users/x_aleho/video-diffusion/data/generated/final_evals/phenotype_ms/ms-LOW"
#  "/proj/aicell/users/x_aleho/video-diffusion/data/generated/final_evals/phenotype_pr/pr-HIGH"
#  "/proj/aicell/users/x_aleho/video-diffusion/data/generated/final_evals/phenotype_pr/pr-LOW"
#  "/proj/aicell/users/x_aleho/video-diffusion/data/generated/final_evals/prompt_cc/cc-HIGH"
#  "/proj/aicell/users/x_aleho/video-diffusion/data/generated/final_evals/prompt_cc/cc-LOW"
#  "/proj/aicell/users/x_aleho/video-diffusion/data/generated/final_evals/prompt_frames/frames-HIGH"
#  "/proj/aicell/users/x_aleho/video-diffusion/data/generated/final_evals/prompt_ms/ms-HIGH"
#  "/proj/aicell/users/x_aleho/video-diffusion/data/generated/final_evals/prompt_ms/ms-LOW"
#  "/proj/aicell/users/x_aleho/video-diffusion/data/generated/final_evals/prompt_pr/pr-HIGH"
#  "/proj/aicell/users/x_aleho/video-diffusion/data/generated/final_evals/prompt_pr/pr-LOW"
#  "/proj/aicell/users/x_aleho/video-diffusion/data/generated/final_evals/uncond/frames81"
 "/proj/aicell/users/x_aleho/video-diffusion/data/generated/final_evals/uncond/frames129"
)

mkdir -p "${LOG_DIR}"

#############################
# LAUNCH JOBS IN PARALLEL  #
#############################

for DIR in "${DIRS[@]}"; do
  BASENAME=$(basename "$DIR")  # e.g. 'ms-LOW'
  
  # Get the parent directory name
  PARENT_DIR=$(basename "$(dirname "$DIR")")  # e.g. 'prompt_ms'
  
  # Create combined output directory name 
  COMBINED_NAME="${PARENT_DIR}_${BASENAME}"
  
  OUTPUT_DIR="${OUTPUT_BASE_DIR}/${COMBINED_NAME}"
  LOG_FILE="${LOG_DIR}/${COMBINED_NAME}.log"

  echo "Launching mask creation for: ${DIR}"
  echo "Output to: ${OUTPUT_DIR}"
  
  python "${SCRIPT_PATH}" \
    --input_dir "${DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    > "${LOG_FILE}" 2>&1 &

done

# Wait for all background jobs to finish
wait

echo "All create_masks.py jobs have completed."
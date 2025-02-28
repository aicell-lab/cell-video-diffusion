#!/bin/bash
#SBATCH -A berzelius-2025-23    # Your project/account
#SBATCH --gpus=1 -C "thin"        # Number of GPUs needed
#SBATCH -t 2-00:00:00            # Time limit (e.g. 1 day)
#SBATCH -J real_distributions         # Job name
#SBATCH -o logs/%x_%j.out        # Standard output log
#SBATCH -e logs/%x_%j.err        # Standard error log

# run_distributions.sh
#
# Example usage:
#   bash run_distributions.sh
#
# This script:
# Calls compute_distribution.py for a single directory.

SCRIPT_PATH="compute_distributions.py"
INPUT_DIR="/proj/aicell/users/x_aleho/video-diffusion/data/processed/idr0013/val/checkpoint-900-val"
BASENAME=$(basename "$INPUT_DIR")
OUTPUT_BASE_DIR="results"
OUTPUT_DIR="${OUTPUT_BASE_DIR}/${BASENAME}"
LOG_DIR="logs"


mkdir -p "${LOG_DIR}"

echo "Launching distribution computation for: ${INPUT_DIR}"
python "${SCRIPT_PATH}" \
  --input_dir "${INPUT_DIR}" \
  --output_dir "${OUTPUT_DIR}"
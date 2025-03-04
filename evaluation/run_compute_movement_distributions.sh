#!/usr/bin/env bash

# ensure that if the script is interrupted, all background jobs are killed
trap "echo 'Received SIGINT. Killing background jobs...'; kill 0" SIGINT

# run_compute_movement_distributions.sh
#
# Example usage:
#   bash run_compute_movement_distributions.sh
#
# This script:
# 1) Iterates through a list of directories containing .npy mask files.
# 2) Calls compute_movement_distributions.py for each directory in parallel.
# 3) Saves logs for each run to logs/ subdirectory.

##############################
# USER-DEFINED PATHS/PARAMS #
##############################

SCRIPT_PATH="compute_movement_distributions.py"
OUTPUT_BASE_DIR="results"
LOG_DIR="logs/movement"

DIRS=(
  # "/proj/aicell/users/x_aleho/video-diffusion/evaluation/masks_output/i2v_r64_150"
  # "/proj/aicell/users/x_aleho/video-diffusion/evaluation/masks_output/i2v_r64_250"
  # "/proj/aicell/users/x_aleho/video-diffusion/evaluation/masks_output/i2v_r64_500"
  # "/proj/aicell/users/x_aleho/video-diffusion/evaluation/masks_output/i2v_r64_750"
  # "/proj/aicell/users/x_aleho/video-diffusion/evaluation/masks_output/i2v_r64_900"
  # "/proj/aicell/users/x_aleho/video-diffusion/evaluation/masks_output/i2v_r128_250"
  # "/proj/aicell/users/x_aleho/video-diffusion/evaluation/masks_output/i2v_r128_500"
  # "/proj/aicell/users/x_aleho/video-diffusion/evaluation/masks_output/i2v_r128_750"
  # "/proj/aicell/users/x_aleho/video-diffusion/evaluation/masks_output/i2v_r128_900"
  # "/proj/aicell/users/x_aleho/video-diffusion/evaluation/masks_output/i2v_r256_150"
  # "/proj/aicell/users/x_aleho/video-diffusion/evaluation/masks_output/i2v_r256_250"
  # "/proj/aicell/users/x_aleho/video-diffusion/evaluation/masks_output/i2v_r256_375"
  # "/proj/aicell/users/x_aleho/video-diffusion/evaluation/masks_output/i2v_r256_750"
  # "/proj/aicell/users/x_aleho/video-diffusion/evaluation/masks_output/i2v_r256_900"
  # "/proj/aicell/users/x_aleho/video-diffusion/evaluation/masks_output/checkpoint-900-val"
  # "/proj/aicell/users/x_aleho/video-diffusion/evaluation/masks_output/i2v_baseline"
  "/proj/aicell/users/x_aleho/video-diffusion/evaluation/masks_output/IDR0013-FILTERED-Test-ms-LOW"
  "/proj/aicell/users/x_aleho/video-diffusion/evaluation/masks_output/IDR0013-FILTERED-Test-ms-MED"
  "/proj/aicell/users/x_aleho/video-diffusion/evaluation/masks_output/IDR0013-FILTERED-Test-ms-HIGH"
)

#############################
# CREATE LOG DIR IF NEEDED #
#############################

mkdir -p "${LOG_DIR}"

#############################
# LAUNCH JOBS IN PARALLEL  #
#############################

for DIR in "${DIRS[@]}"; do
  BASENAME=$(basename "$DIR")         # e.g. 'i2v_r64_500'
  OUTPUT_DIR="${OUTPUT_BASE_DIR}/${BASENAME}"
  LOG_FILE="${LOG_DIR}/${BASENAME}.log"

  echo "Launching movement distribution computation for: ${DIR}"
  python "${SCRIPT_PATH}" \
    --input_dir "${DIR}" \
    --output_dir "${OUTPUT_DIR}" \
    > "${LOG_FILE}" 2>&1 &

  # The '>' redirects stdout to the log file, '2>&1' redirects stderr there as well
  # '&' indicates run this job in the background
done

# Wait for all background jobs to finish
wait

echo "All compute_movement_distributions.py jobs have completed." 
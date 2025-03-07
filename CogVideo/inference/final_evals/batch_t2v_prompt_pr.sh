#!/usr/bin/env bash
#SBATCH -A berzelius-2025-23
#SBATCH --gpus=1 -C "thin"
#SBATCH -t 1-00:00:00
#SBATCH -J batch_t2v_prompt_pr_ARG1
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# Get HIGH or LOW from command line argument
CONDITION=$1
if [[ "$CONDITION" != "HIGH" && "$CONDITION" != "LOW" ]]; then
  echo "Error: First argument must be either HIGH or LOW"
  exit 1
fi

echo "Generating videos for proliferation condition: $CONDITION"
echo "Loading model once and generating 15 videos with seeds 1-15"

module load Mambaforge/23.3.1-1-hpc1-bdist
conda activate /proj/aicell/users/x_aleho/conda_envs/cogvideo

# Create the needed directory structure
mkdir -p configs

# Run the batch generator with the appropriate config
python batch_generator.py --config configs/prompt_pr_${CONDITION}.json 
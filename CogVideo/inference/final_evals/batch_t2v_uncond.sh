#!/usr/bin/env bash
#SBATCH -A berzelius-2025-23
#SBATCH --gpus=1 -C "thin"
#SBATCH -t 1-00:00:00
#SBATCH -J batch_t2v_uncond
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err


echo "Generating videos for unconditioned model"
echo "Loading model once and generating 15 videos with seeds 1-15"

module load Mambaforge/23.3.1-1-hpc1-bdist
conda activate /proj/aicell/users/x_aleho/conda_envs/cogvideo

# Create the needed directory structure
mkdir -p configs

# Run the batch generator with the appropriate config
python batch_generator.py --config configs/uncond_frames.json 
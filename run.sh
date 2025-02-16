#!/bin/bash
#SBATCH -A berzelius-2025-23    # Your project/account
#SBATCH --gpus=1 -C "fat"        # Number of GPUs needed
#SBATCH -t 1-00:00:00            # Time limit (e.g. 1 day)
#SBATCH --cpus-per-gpu=16        # CPU cores per GPU (adjust as needed)
#SBATCH --mem=128G               # Total memory (adjust as needed)
#SBATCH -J sd2_mca_train         # Job name
#SBATCH -o logs/%x_%j.out        # Standard output log
#SBATCH -e logs/%x_%j.err        # Standard error log

module load Mambaforge/23.3.1-1-hpc1-bdist

conda activate /proj/aicell/users/x_aleho/conda_envs/cell-video-diffusion

echo "Using GPU(s): $CUDA_VISIBLE_DEVICES"
nvidia-smi

accelerate launch src/train_sd2.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-base" \
  --train_data_dir="data/processed/mca_frames_256/train" \
  --output_dir="./models/sd2-mca-frame-embeddings" \
  --num_train_epochs=100 \
  --train_batch_size=16 \
  --learning_rate=1e-5 \
  --wandb_project="sd2_cell_finetune" \
  --wandb_run_name="mca_run_frame_embeddings" \
  --log_frequency=20 \
  --max_frame_idx=40 \
  --sample_frames 0 9 19 29 39
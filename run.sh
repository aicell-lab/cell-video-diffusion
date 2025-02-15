#!/bin/bash
#SBATCH -A berzelius-2025-23    # Your project/account
#SBATCH --gpus=1 -C "fat"        # Number of GPUs needed
#SBATCH -t 1-00:00:00            # Time limit (e.g. 1 day)
#SBATCH --cpus-per-gpu=16        # CPU cores per GPU (adjust as needed)
#SBATCH --mem=128G               # Total memory (adjust as needed)
#SBATCH -J sd2_mca_train         # Job name
#SBATCH -o logs/%x_%j.out        # Standard output log
#SBATCH -e logs/%x_%j.err        # Standard error log

# 1) Load any required modules
module load Mambaforge/23.3.1-1-hpc1-bdist

# 2) Activate your conda environment
conda activate /proj/aicell/users/x_aleho/conda_envs/cell-video-diffusion

# 3) (Optional) Print some debug info
echo "Using GPU(s): $CUDA_VISIBLE_DEVICES"
nvidia-smi

# 4) Run your training script with accelerate
accelerate launch src/train_sd2_A.py \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-base" \
    --train_data_dir="data/processed/mca_frames_256/train" \
    --output_dir="./models/sd2-mca-finetuned-frames" \
    --num_train_epochs=10 \
    --train_batch_size=4 \
    --learning_rate=1e-5 \
    --wandb_project="sd2_cell_finetune" \
    --wandb_run_name="mca_run_frames" \
    --log_frequency=20 \
    --sample_prompts "a microscopy image of a cell at frame 0" "a microscopy image of a cell at frame 9" "a microscopy image of a cell at frame 19" "a microscopy image of a cell at frame 39" "a microscopy image of a cell"

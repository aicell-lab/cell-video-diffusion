#!/bin/bash
#SBATCH -A berzelius-2025-23    # Your project/account
#SBATCH --gpus=1 -C "fat"        # Number of GPUs needed
#SBATCH -t 1-00:00:00            # Time limit (e.g. 1 day)
#SBATCH --cpus-per-gpu=16        # CPU cores per GPU (adjust as needed)
#SBATCH --mem=128G               # Total memory (adjust as needed)
#SBATCH -J cogvideo_i2v_train         # Job name
#SBATCH -o logs/%x_%j.out        # Standard output log
#SBATCH -e logs/%x_%j.err        # Standard error log

module load Mambaforge/23.3.1-1-hpc1-bdist

conda activate /proj/aicell/users/x_aleho/conda_envs/cogvideo


# Prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false

# Model Configuration
MODEL_ARGS=(
    --model_path "../models/CogVideoX1.5-5B-I2V"
    --model_name "cogvideox1.5-i2v"  # ["cogvideox-i2v"]
    --model_type "i2v"
    --training_type "lora"
)

# Output Configuration
OUTPUT_ARGS=(
    --output_dir "../models/loras/idr0013-i2v-overfit"
    --report_to "wandb"
)

# Data Configuration
DATA_ARGS=(
    --data_root "./IDR0013-VidGene-1"
    --caption_column "prompts.txt"
    --video_column "videos.txt"
    # --image_column "images.txt"  # comment this line will use first frame of video as image conditioning
    --train_resolution "81x768x1360"  # (frames x height x width), frames should be 8N+1
)

# Training Configuration
TRAIN_ARGS=(
    --train_epochs 10 # number of training epochs
    --seed 42 # random seed
    --batch_size 1
    --gradient_accumulation_steps 1
    --mixed_precision "bf16"  # ["no", "fp16"] # Only CogVideoX-2B supports fp16 training
    # --rank 256
    # --lora_alpha 256
)

# System Configuration
SYSTEM_ARGS=(
    --num_workers 8
    --pin_memory True
    --nccl_timeout 1800
)

# Checkpointing Configuration
CHECKPOINT_ARGS=(
    --checkpointing_steps 1 # save checkpoint every x steps
    --checkpointing_limit 2 # maximum number of checkpoints to keep, after which the oldest one is deleted
    # --resume_from_checkpoint "../models/loras/idr0013-i2v-50/checkpoint-250"  # if you want to resume from a checkpoint, otherwise, comment this line
)

# Validation Configuration
VALIDATION_ARGS=(
    --do_validation true  # ["true", "false"]
    --validation_dir "./IDR0013-VidGene-Val-1"
    --validation_steps 1  # should be multiple of checkpointing_steps
    --validation_prompts "prompts.txt"
    --validation_images "images.txt"
    --validation_videos "videos.txt" # is this handled correctly?
    # --gen_fps 16
)

# Combine all arguments and launch training
accelerate launch train.py \
    "${MODEL_ARGS[@]}" \
    "${OUTPUT_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${SYSTEM_ARGS[@]}" \
    "${CHECKPOINT_ARGS[@]}" \
    "${VALIDATION_ARGS[@]}"

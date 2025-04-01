#!/bin/bash
#SBATCH -A berzelius-2025-23    # Your project/account
#SBATCH --gpus=4 -C "fat"        # Number of GPUs needed
#SBATCH -t 2-00:00:00            # Time limit (e.g. 1 day)
#SBATCH -J t2v_sft               # Job name
#SBATCH -o logs/%x_%j.out        # Standard output log
#SBATCH -e logs/%x_%j.err        # Standard error log

module load Mambaforge/23.3.1-1-hpc1-bdist

conda activate /proj/aicell/users/x_aleho/conda_envs/cogvideo

DATASET_NAME="IDR0013-FILTERED-uncond"

# Prevent tokenizer parallelism issues
export TOKENIZERS_PARALLELISM=false

# Model Configuration
MODEL_ARGS=(
    --model_path "../models/CogVideoX1.5-5B"
    --model_name "cogvideox1.5-t2v"
    --model_type "t2v"
    --training_type "sft"
)

# Output Configuration
OUTPUT_ARGS=(
    --output_dir "../models/sft/${DATASET_NAME}-t2v-uncond-final"
    --report_to "wandb"
)

# Data Configuration
DATA_ARGS=(
    --data_root "../../data/ready/${DATASET_NAME}"
    --caption_column "prompts.txt"
    --video_column "videos.txt"
    --train_resolution "81x768x1360"  # (frames x height x width), frames should be 8N+1
)

# Training Configuration
TRAIN_ARGS=(
    --train_epochs 3 # number of training epochs
    --seed 42 # random seed
    --batch_size 1
    --gradient_accumulation_steps 1
    --mixed_precision "bf16"  # ["no", "fp16"] Only CogVideoX-2B supports fp16 training
    ########################################################################
)

# System Configuration
SYSTEM_ARGS=(
    --num_workers 8
    --pin_memory True
    --nccl_timeout 1800
)

# Checkpointing Configuration
CHECKPOINT_ARGS=(
    --checkpointing_steps 200 # save checkpoint every x steps
    --checkpointing_limit 50 # maximum number of checkpoints to keep, after which the oldest one is deleted
    # --resume_from_checkpoint "/path/to/checkpoint"  # if you want to resume from a checkpoint, otherwise, comment this line
)

# Validation Configuration
VALIDATION_ARGS=(
    --do_validation true  # ["true", "false"]
    --validation_dir "../../data/ready/${DATASET_NAME}-Val"
    --validation_steps 200  # should be multiple of checkpointing_steps
    --validation_prompts "prompts.txt"
    --gen_fps 10
)

# Combine all arguments and launch training with ZeRO optimization
accelerate launch --main_process_port=29505 --config_file accelerate_config1.yaml train.py \
    "${MODEL_ARGS[@]}" \
    "${OUTPUT_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${TRAIN_ARGS[@]}" \
    "${SYSTEM_ARGS[@]}" \
    "${CHECKPOINT_ARGS[@]}" \
    "${VALIDATION_ARGS[@]}"

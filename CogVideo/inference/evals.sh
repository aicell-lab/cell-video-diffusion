#!/usr/bin/env bash

# Create logs directory if it doesn't exist
mkdir -p logs

# Submit baseline job (no LoRA)
# echo "Submitting baseline job (no LoRA)..."
# sbatch baseline.sh
# echo "Submitted baseline job"

# Submit jobs for each model type and checkpoint
# echo "Submitting LoRA rank 64 jobs..."
# for CHECKPOINT in 250 500 750 900 ; do
#     sbatch i2v_r64.sh $CHECKPOINT
#     echo "Submitted job for rank 64, checkpoint $CHECKPOINT"
# done

# echo "Submitting LoRA rank 128 jobs..."
# for CHECKPOINT in 250 500 750 900; do
#     sbatch i2v_r128.sh $CHECKPOINT
#     echo "Submitted job for rank 128, checkpoint $CHECKPOINT"
# done

# echo "Submitting LoRA rank 256 jobs..."
# for CHECKPOINT in 250 375 750 900; do
#     sbatch i2v_r256.sh $CHECKPOINT
#     echo "Submitted job for rank 256, checkpoint $CHECKPOINT"
# done

echo "Submitting full finetune jobs..."
for CHECKPOINT in 250 500 750 900; do
    sbatch i2v_sft.sh $CHECKPOINT
    echo "Submitted job for SFT, checkpoint $CHECKPOINT"
done

echo "All evaluation jobs have been submitted!"

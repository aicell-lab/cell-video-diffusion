#!/usr/bin/env bash

# Create logs directory if it doesn't exist
mkdir -p logs

Submit jobs for each model type and checkpoint
echo "Submitting LoRA rank 64 jobs..."
for CHECKPOINT in 150 250 400; do
    sbatch i2v_r64.sh $CHECKPOINT
    echo "Submitted job for rank 64, checkpoint $CHECKPOINT"
done

echo "Submitting LoRA rank 128 jobs..."
for CHECKPOINT in 250 400 500; do
    sbatch i2v_r128.sh $CHECKPOINT
    echo "Submitted job for rank 128, checkpoint $CHECKPOINT"
done

echo "Submitting LoRA rank 256 jobs..."
for CHECKPOINT in 150 250 400; do
    sbatch i2v_r256.sh $CHECKPOINT
    echo "Submitted job for rank 256, checkpoint $CHECKPOINT"
done

echo "Submitting full finetune jobs..."
for CHECKPOINT in 100 200 300; do
    sbatch i2v_sft.sh $CHECKPOINT
    echo "Submitted job for SFT, checkpoint $CHECKPOINT"
done

echo "All evaluation jobs have been submitted!"

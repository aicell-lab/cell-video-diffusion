#!/bin/bash
accelerate launch src/train_sd2_A.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-base" \
  --train_data_dir="data/processed/mca_frame0_256/train" \
  --output_dir="./sd2-mca-finetuned" \
  --num_train_epochs=10 \
  --train_batch_size=4 \
  --learning_rate=1e-5 \
  --wandb_project="sd2_cell_finetune" \
  --wandb_run_name="mca_run1" \
  --log_frequency=20 \
  --sample_prompts "a microscopy image of a cell"

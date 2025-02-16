#!/usr/bin/env bash

accelerate launch src/train_sd2.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-base" \
  --train_data_dir="data/processed/mca_frames_256/train" \
  --output_dir="./models/sd2-mca-frame-embeddings" \
  --num_train_epochs=10 \
  --train_batch_size=4 \
  --learning_rate=1e-5 \
  --wandb_project="sd2_cell_finetune" \
  --wandb_run_name="mca_run_frame_embeddings" \
  --log_frequency=20 \
  --max_frame_idx=40 \
  --sample_frames 0 9 19 29 39

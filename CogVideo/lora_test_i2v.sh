#!/usr/bin/env bash
#SBATCH -A berzelius-2025-23
#SBATCH --gpus=1 -C "thin"
#SBATCH -t 1-00:00:00
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=128G
#SBATCH -J i2v_no_prompt_test
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

module load Mambaforge/23.3.1-1-hpc1-bdist
conda activate /proj/aicell/users/x_aleho/conda_envs/cogvideo

# Base model & LoRA
MODEL_PATH="./models/CogVideoX1.5-5B-I2V"
LORA_PATH="./models/loras/idr0013-i2v-50/checkpoint-250"
GENERATE_TYPE="i2v"

# Inference params
NUM_STEPS=30
GUIDANCE_SCALE=6.0
NUM_FRAMES=81  # 8N+1 for CogVideoX1.5
FPS=10
OUTDIR="./test_generations/i2v_no_prompt"
mkdir -p "$OUTDIR"

# List your image paths here
IMAGES=(
"/proj/aicell/users/x_aleho/video-diffusion/data/processed/idr0013/first_frames/00005_01.png"
"/proj/aicell/users/x_aleho/video-diffusion/data/processed/idr0013/first_frames/00012_01.png"
"/proj/aicell/users/x_aleho/video-diffusion/data/processed/idr0013/first_frames/00019_01.png"
)

for i in "${!IMAGES[@]}"; do
  IMAGE_PATH="${IMAGES[$i]}"
  BASENAME=$(basename "$IMAGE_PATH" .png)  # e.g. "00005_01"

  ################################################################
  # 1) No LoRA
  ################################################################
  OUTPUT_NO_LORA="${OUTDIR}/${BASENAME}_noLORA.mp4"
  echo "=== Generating with NO LoRA ==="
  echo "Image: $IMAGE_PATH"
  echo "Output: $OUTPUT_NO_LORA"

  python inference/cli_demo.py \
    --prompt "" \
    --model_path "$MODEL_PATH" \
    --image_or_video_path "$IMAGE_PATH" \
    --generate_type "$GENERATE_TYPE" \
    --num_inference_steps "$NUM_STEPS" \
    --guidance_scale "$GUIDANCE_SCALE" \
    --num_frames "$NUM_FRAMES" \
    --fps "$FPS" \
    --output_path "$OUTPUT_NO_LORA"

  ################################################################
  # 2) With IDR0013 I2V LoRA
  ################################################################
  OUTPUT_LORA="${OUTDIR}/${BASENAME}_LOIRA.mp4"
  echo "=== Generating WITH IDR0013 I2V LoRA ==="
  echo "Image: $IMAGE_PATH"
  echo "Output: $OUTPUT_LORA"

  python inference/cli_demo.py \
    --prompt "" \
    --model_path "$MODEL_PATH" \
    --lora_path "$LORA_PATH" \
    --image_or_video_path "$IMAGE_PATH" \
    --generate_type "$GENERATE_TYPE" \
    --num_inference_steps "$NUM_STEPS" \
    --guidance_scale "$GUIDANCE_SCALE" \
    --num_frames "$NUM_FRAMES" \
    --fps "$FPS" \
    --output_path "$OUTPUT_LORA"
done

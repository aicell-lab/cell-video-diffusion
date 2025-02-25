#!/usr/bin/env bash
#SBATCH -A berzelius-2025-23
#SBATCH --gpus=1 -C "thin"
#SBATCH -t 1-00:00:00
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=128G
#SBATCH -J i2v_fid_r256_300
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

module load Mambaforge/23.3.1-1-hpc1-bdist
conda activate /proj/aicell/users/x_aleho/conda_envs/cogvideo

# -------------------------------
# MODEL & LORA
# -------------------------------
MODEL_PATH="../models/CogVideoX1.5-5B-I2V"
LORA_PATH="../models/loras/r256/checkpoint-300"
GENERATE_TYPE="i2v"

# -------------------------------
# ONE PROMPT, MULTIPLE IMAGES
# -------------------------------
# Single prompt for all generations
PROMPT="Time-lapse microscopy video with medium migration speed, high migration distance, and high proliferation."

# Base directory for input images
IMAGE_BASE_DIR="/proj/aicell/users/x_aleho/video-diffusion/data/processed/idr0013/first_frames"

# Output folder
OUTDIR="../test_generations/i2v_fid_r256_300"
mkdir -p "$OUTDIR"

# Fixed parameters for all generations
STEPS=30
SCALE=8
NUM_FRAMES=81
FPS=10

# Loop through image numbers from 1 to 25
for i in $(seq -f "%05g" 1 25); do
  # Construct the image path using the pattern
  # goes from LT0003_15-00001_01.png to LT0003_15-00025_01.png
  IMAGE_PATH="${IMAGE_BASE_DIR}/LT0003_15-${i}_01.png"
  
  # Get basename for the image
  BASENAME=$(basename "$IMAGE_PATH" .png)
  
  echo "=================================================="
  echo "Generating video for image: $BASENAME"
  echo "Prompt: $PROMPT"
  echo "Image:  $IMAGE_PATH"
  
  # Construct output filename
  OUTPUT_PATH="${OUTDIR}/${BASENAME}_S${STEPS}_G${SCALE}_F${NUM_FRAMES}_FPS${FPS}.mp4"
  
  echo "Output => ${OUTPUT_PATH}"
  echo "--------------------------------------------------"
  
  python cli_demo.py \
    --prompt "$PROMPT" \
    --model_path "$MODEL_PATH" \
    --lora_path "$LORA_PATH" \
    --image_or_video_path "$IMAGE_PATH" \
    --generate_type "$GENERATE_TYPE" \
    --num_inference_steps "$STEPS" \
    --guidance_scale "$SCALE" \
    --num_frames "$NUM_FRAMES" \
    --fps "$FPS" \
    --output_path "$OUTPUT_PATH"
done

echo "All done! Created 25 videos in $OUTDIR"

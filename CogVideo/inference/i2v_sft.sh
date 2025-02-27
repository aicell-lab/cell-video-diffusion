#!/usr/bin/env bash
#SBATCH -A berzelius-2025-23
#SBATCH --gpus=1 -C "thin"
#SBATCH -t 1-00:00:00
#SBATCH -J i2v_gen_sft
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# Get checkpoint from command line argument
CHECKPOINT=$1
echo "Using checkpoint: $CHECKPOINT"

module load Mambaforge/23.3.1-1-hpc1-bdist
conda activate /proj/aicell/users/x_aleho/conda_envs/cogvideo

# -------------------------------
# MODEL (fully finetuned, no LoRA)
# -------------------------------
BASE_MODEL_PATH="../models/CogVideoX1.5-5B-I2V"
WEIGHTS_PATH="../models/sft/IDR0013-10plates-i2v-1/checkpoint-${CHECKPOINT}/fp32_model"
GENERATE_TYPE="i2v"

# -------------------------------
# ONE PROMPT, MULTIPLE IMAGES
# -------------------------------
# Single prompt for all generations
PROMPT="<ALEXANDER> Time-lapse microscopy video with high proliferation."

# Base directory for input images
IMAGE_BASE_DIR="/proj/aicell/users/x_aleho/video-diffusion/data/processed/idr0013/val/first_frames"

# Output folder
OUTDIR="./test_generations/i2v_sft_${CHECKPOINT}"
mkdir -p "$OUTDIR"

# Fixed parameters for all generations
STEPS=50
SCALE=8
NUM_FRAMES=81
FPS=10

# Loop through image numbers from 1 to 10
for i in $(seq -f "%05g" 1 10); do
  # Construct the image path using the pattern
  IMAGE_PATH="${IMAGE_BASE_DIR}/LT0004_11-${i}_01.png"
  
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
    --base_model_path "$BASE_MODEL_PATH" \
    --weights_path "$WEIGHTS_PATH" \
    --image_or_video_path "$IMAGE_PATH" \
    --generate_type "$GENERATE_TYPE" \
    --num_inference_steps "$STEPS" \
    --guidance_scale "$SCALE" \
    --num_frames "$NUM_FRAMES" \
    --fps "$FPS" \
    --output_path "$OUTPUT_PATH"
done

echo "All done! Created 10 videos in $OUTDIR" 
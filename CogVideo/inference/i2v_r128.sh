#!/usr/bin/env bash
#SBATCH -A berzelius-2025-23
#SBATCH --gpus=1 -C "thin"
#SBATCH -t 1-00:00:00
#SBATCH -J i2v_gen_r128
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# Get checkpoint from command line argument
CHECKPOINT=$1
echo "Using checkpoint: $CHECKPOINT"

module load Mambaforge/23.3.1-1-hpc1-bdist
conda activate /proj/aicell/users/x_aleho/conda_envs/cogvideo

# -------------------------------
# MODEL & LORA
# -------------------------------
MODEL_PATH="../models/CogVideoX1.5-5B-I2V"
LORA_PATH="../models/loras/IDR0013-10plates-i2v-r128-a64/checkpoint-${CHECKPOINT}"
GENERATE_TYPE="i2v"

# -------------------------------
# EVALUATION SETUP: 5 IMAGES, SINGLE SEED
# -------------------------------
# Array of image paths
IMAGE_PATHS=(
  /proj/aicell/users/x_aleho/video-diffusion/data/processed/idr0013/val/first_frames/LT0004_47-00139_01.png
  /proj/aicell/users/x_aleho/video-diffusion/data/processed/idr0013/val/first_frames/LT0004_47-00327_01.png
  /proj/aicell/users/x_aleho/video-diffusion/data/processed/idr0013/val/first_frames/LT0004_47-00301_01.png
  /proj/aicell/users/x_aleho/video-diffusion/data/processed/idr0013/val/first_frames/LT0004_47-00358_01.png
  /proj/aicell/users/x_aleho/video-diffusion/data/processed/idr0013/val/first_frames/LT0004_47-00149_01.png
  /proj/aicell/users/x_aleho/video-diffusion/data/processed/idr0013/val/first_frames/LT0004_47-00271_01.png
  /proj/aicell/users/x_aleho/video-diffusion/data/processed/idr0013/val/first_frames/LT0004_47-00008_01.png
  /proj/aicell/users/x_aleho/video-diffusion/data/processed/idr0013/val/first_frames/LT0004_11-00177_01.png
  /proj/aicell/users/x_aleho/video-diffusion/data/processed/idr0013/val/first_frames/LT0004_11-00376_01.png
  /proj/aicell/users/x_aleho/video-diffusion/data/processed/idr0013/val/first_frames/LT0004_11-00008_01.png
)

# Corresponding prompts for each image
PROMPTS=(
  "<ALEXANDER> Time-lapse microscopy video with medium proliferation."
  "<ALEXANDER> Time-lapse microscopy video with low proliferation."
  "<ALEXANDER> Time-lapse microscopy video with low proliferation."
  "<ALEXANDER> Time-lapse microscopy video with low proliferation."
  "<ALEXANDER> Time-lapse microscopy video with high proliferation."
  "<ALEXANDER> Time-lapse microscopy video with low proliferation."
  "<ALEXANDER> Time-lapse microscopy video with low proliferation."
  "<ALEXANDER> Time-lapse microscopy video with high proliferation."
  "<ALEXANDER> Time-lapse microscopy video with high proliferation."
  "<ALEXANDER> Time-lapse microscopy video with high proliferation."
)

# Use a single seed for all generations
SEED=9

# Output folder
OUTDIR="./test_generations_gpt4o_mtscore/i2v_r128_${CHECKPOINT}"
mkdir -p "$OUTDIR"

# Fixed parameters for all generations
STEPS=50
SCALE=8
NUM_FRAMES=81
FPS=10

# Loop through each image-prompt pair
for i in {0..9}; do
  IMAGE_PATH="${IMAGE_PATHS[$i]}"
  PROMPT="${PROMPTS[$i]}"
  
  # Get basename for the image
  BASENAME=$(basename "$IMAGE_PATH" .png)
  
  echo "=================================================="
  echo "Processing image $((i+1))/10: $BASENAME"
  echo "Prompt: $PROMPT"
  echo "Using seed: $SEED"
  
  # Construct output filename
  OUTPUT_PATH="${OUTDIR}/${BASENAME}_seed${SEED}_S${STEPS}_G${SCALE}_F${NUM_FRAMES}_FPS${FPS}.mp4"
  
  echo "Output => ${OUTPUT_PATH}"
  
  python cli_demo.py \
    --prompt "$PROMPT" \
    --seed "$SEED" \
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

echo "All done! Created 10 videos (10 images x 1 seed) in $OUTDIR" 
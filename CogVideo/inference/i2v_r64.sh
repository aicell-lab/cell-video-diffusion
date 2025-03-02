#!/usr/bin/env bash
#SBATCH -A berzelius-2025-23
#SBATCH --gpus=1 -C "thin"
#SBATCH -t 1-00:00:00
#SBATCH -J i2v_gen_r64
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
LORA_PATH="../models/loras/IDR0013-10plates-i2v-r64-a32/checkpoint-${CHECKPOINT}"
GENERATE_TYPE="i2v"

# -------------------------------
# EVALUATION SETUP: 10 IMAGES, SINGLE SEED
# -------------------------------
# Array of image paths
IMAGE_PATHS=(
  /proj/aicell/users/x_aleho/video-diffusion/data/processed/idr0013/first_frames/LT0004_06-00083_01.png
  /proj/aicell/users/x_aleho/video-diffusion/data/processed/idr0013/first_frames/LT0002_51-00053_01.png
  /proj/aicell/users/x_aleho/video-diffusion/data/processed/idr0013/first_frames/LT0004_06-00189_01.png
  /proj/aicell/users/x_aleho/video-diffusion/data/processed/idr0013/first_frames/LT0004_06-00093_01.png
  /proj/aicell/users/x_aleho/video-diffusion/data/processed/idr0013/first_frames/LT0003_02-00171_01.png
  /proj/aicell/users/x_aleho/video-diffusion/data/processed/idr0013/first_frames/LT0001_02-00313_01.png
  /proj/aicell/users/x_aleho/video-diffusion/data/processed/idr0013/first_frames/LT0001_02-00266_01.png
  /proj/aicell/users/x_aleho/video-diffusion/data/processed/idr0013/first_frames/LT0001_09-00073_01.png
  /proj/aicell/users/x_aleho/video-diffusion/data/processed/idr0013/first_frames/LT0001_09-00195_01.png
  /proj/aicell/users/x_aleho/video-diffusion/data/processed/idr0013/first_frames/LT0004_06-00376_01.png
# /proj/aicell/users/x_aleho/video-diffusion/data/processed/idr0013/first_frames/LT0002_02-00263_01.png
# /proj/aicell/users/x_aleho/video-diffusion/data/processed/idr0013/first_frames/LT0002_51-00072_01.png
# /proj/aicell/users/x_aleho/video-diffusion/data/processed/idr0013/first_frames/LT0004_06-00239_01.png
# /proj/aicell/users/x_aleho/video-diffusion/data/processed/idr0013/first_frames/LT0001_02-00122_01.png
# /proj/aicell/users/x_aleho/video-diffusion/data/processed/idr0013/first_frames/LT0003_02-00210_01.png
# /proj/aicell/users/x_aleho/video-diffusion/data/processed/idr0013/first_frames/LT0002_02-00241_01.png
# /proj/aicell/users/x_aleho/video-diffusion/data/processed/idr0013/first_frames/LT0002_24-00314_01.png
# /proj/aicell/users/x_aleho/video-diffusion/data/processed/idr0013/first_frames/LT0002_51-00111_01.png
# /proj/aicell/users/x_aleho/video-diffusion/data/processed/idr0013/first_frames/LT0004_06-00053_01.png
# /proj/aicell/users/x_aleho/video-diffusion/data/processed/idr0013/first_frames/LT0002_02-00086_01.png
)

# Corresponding prompts for each image
PROMPTS=(
  "<ALEXANDER> Time-lapse microscopy video with medium proliferation."
  "<ALEXANDER> Time-lapse microscopy video with medium proliferation."
  "<ALEXANDER> Time-lapse microscopy video with low proliferation."
  "<ALEXANDER> Time-lapse microscopy video with medium proliferation."
  "<ALEXANDER> Time-lapse microscopy video with medium proliferation."
  "<ALEXANDER> Time-lapse microscopy video with high proliferation."
  "<ALEXANDER> Time-lapse microscopy video with high proliferation."
  "<ALEXANDER> Time-lapse microscopy video with medium proliferation."
  "<ALEXANDER> Time-lapse microscopy video with medium proliferation."
  "<ALEXANDER> Time-lapse microscopy video with high proliferation."
  # "<ALEXANDER> Time-lapse microscopy video with medium proliferation."
  # "<ALEXANDER> Time-lapse microscopy video with high proliferation."
  # "<ALEXANDER> Time-lapse microscopy video with medium proliferation."
  # "<ALEXANDER> Time-lapse microscopy video with high proliferation."
  # "<ALEXANDER> Time-lapse microscopy video with high proliferation."
  # "<ALEXANDER> Time-lapse microscopy video with high proliferation."
  # "<ALEXANDER> Time-lapse microscopy video with high proliferation."
  # "<ALEXANDER> Time-lapse microscopy video with high proliferation."
  # "<ALEXANDER> Time-lapse microscopy video with high proliferation."
  # "<ALEXANDER> Time-lapse microscopy video with medium proliferation."
)

# Use a single seed for all generations
SEED=9

# Output folder
OUTDIR="../../data/generated/test_generations_realval/i2v_r64_${CHECKPOINT}"
mkdir -p "$OUTDIR"

# Fixed parameters for all generations
STEPS=50
SCALE=8
NUM_FRAMES=81
FPS=10

# Loop through each image-prompt pair
# Loop through each image-prompt pair
for i in "${!IMAGE_PATHS[@]}"; do
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
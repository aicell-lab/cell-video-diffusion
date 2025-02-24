#!/usr/bin/env bash
#SBATCH -A berzelius-2025-23
#SBATCH --gpus=1 -C "thin"
#SBATCH -t 1-00:00:00
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=128G
#SBATCH -J i2v_prolif_test
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

module load Mambaforge/23.3.1-1-hpc1-bdist
conda activate /proj/aicell/users/x_aleho/conda_envs/cogvideo

# -----------------------------------------------------
# MODEL & LORA INFO
# -----------------------------------------------------
MODEL_PATH="./models/CogVideoX1.5-5B-I2V"
LORA_PATH="./models/loras/5plates/checkpoint-450"
GENERATE_TYPE="i2v"

# -----------------------------------------------------
# INFERENCE PARAMS
# -----------------------------------------------------
NUM_STEPS=30
GUIDANCE_SCALE=6.0
NUM_FRAMES=81  # 8N+1 for CogVideoX1.5
FPS=10

# -----------------------------------------------------
# OUTPUT CONFIG
# -----------------------------------------------------
OUTDIR="./test_generations/i2v_prompt_test"
mkdir -p "$OUTDIR"

# -----------------------------------------------------
# IMAGE PATHS
# (These remain the same as your original script.)
# -----------------------------------------------------
IMAGES=(
"/proj/aicell/users/x_aleho/video-diffusion/data/processed/idr0013/first_frames/LT0001_09-00374_01.png"
"/proj/aicell/users/x_aleho/video-diffusion/data/processed/idr0013/first_frames/LT0001_12-00088_01.png"
"/proj/aicell/users/x_aleho/video-diffusion/data/processed/idr0013/first_frames/LT0002_24-00089_01.png"
)

# -----------------------------------------------------
# PROMPTS: Fixing migration speed & distance as "medium"
# Varying only proliferation: low, medium, high
# -----------------------------------------------------
PROMPTS=(
"Time-lapse microscopy video with medium migration speed, medium migration distance, and low proliferation."
# "Time-lapse microscopy video with medium migration speed, medium migration distance, and medium proliferation."
"Time-lapse microscopy video with medium migration speed, medium migration distance, and high proliferation."
)

# -----------------------------------------------------
# GENERATION LOOP
# For each IMAGE, for each PROMPT => produce video
# using LoRA for all
# -----------------------------------------------------
for IMAGE_PATH in "${IMAGES[@]}"; do
  BASENAME=$(basename "$IMAGE_PATH" .png)

  for PROMPT_TEXT in "${PROMPTS[@]}"; do
    # Make a safe name for the prompt
    SAFE_PROMPT=$(echo "$PROMPT_TEXT" | grep -o 'low\|medium\|high')
    
    OUTPUT_PATH="${OUTDIR}/${BASENAME}_prolif_${SAFE_PROMPT}.mp4"
    echo "=== Generating with LoRA ==="
    echo "Image: $IMAGE_PATH"
    echo "Prompt: $PROMPT_TEXT"
    echo "Output: $OUTPUT_PATH"
    echo "--------------------------------------------"

    python inference/cli_demo.py \
      --prompt "$PROMPT_TEXT" \
      --model_path "$MODEL_PATH" \
      --lora_path "$LORA_PATH" \
      --image_or_video_path "$IMAGE_PATH" \
      --generate_type "$GENERATE_TYPE" \
      --num_inference_steps "$NUM_STEPS" \
      --guidance_scale "$GUIDANCE_SCALE" \
      --num_frames "$NUM_FRAMES" \
      --fps "$FPS" \
      --output_path "$OUTPUT_PATH"
  done
done
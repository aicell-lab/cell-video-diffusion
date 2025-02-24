#!/usr/bin/env bash
#SBATCH -A berzelius-2025-23
#SBATCH --gpus=1 -C "thin"
#SBATCH -t 1-00:00:00
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=128G
#SBATCH -J i2v_lora_compare
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

module load Mambaforge/23.3.1-1-hpc1-bdist
conda activate /proj/aicell/users/x_aleho/conda_envs/cogvideo

# -----------------------------------------------------
# MODEL & LORA INFO
# -----------------------------------------------------
MODEL_PATH="./models/CogVideoX1.5-5B-I2V"
LORA_PATH="./models/loras/idr0013-i2v-5plates-r128/checkpoint-800"
GENERATE_TYPE="i2v"

# -----------------------------------------------------
# INFERENCE PARAMS
# -----------------------------------------------------
NUM_STEPS=50
GUIDANCE_SCALE=6.0
NUM_FRAMES=81  # 8N+1 for CogVideoX1.5
FPS=10

# -----------------------------------------------------
# OUTPUT CONFIG
# -----------------------------------------------------
OUTDIR="./test_generations/i2v_lora_compare_50steps_800"
mkdir -p "$OUTDIR"

IMAGES=(
"/proj/aicell/users/x_aleho/video-diffusion/data/processed/idr0013/first_frames/LT0001_02-00223_01.png"
# "/proj/aicell/users/x_aleho/video-diffusion/data/processed/idr0013/first_frames/LT0001_09-00374_01.png"
# "/proj/aicell/users/x_aleho/video-diffusion/data/processed/idr0013/first_frames/LT0001_12-00088_01.png"
# "/proj/aicell/users/x_aleho/video-diffusion/data/processed/idr0013/first_frames/LT0002_24-00089_01.png"
)

PROMPT="Time-lapse microscopy video with medium migration speed, high migration distance, and high proliferation."

# -----------------------------------------------------
# GENERATION LOOP
# For each IMAGE, we do 2 runs:
# 1) No LoRA
# 2) With LoRA
# -----------------------------------------------------
for IMAGE_PATH in "${IMAGES[@]}"; do
  BASENAME=$(basename "$IMAGE_PATH" .png)

  # 1) No LoRA
  # OUTPUT_NO_LORA="${OUTDIR}/${BASENAME}_noLORA_high.mp4"
  # echo "=== Generating with NO LoRA ==="
  # echo "Prompt: $PROMPT"
  # echo "Image: $IMAGE_PATH"
  # echo "Output: $OUTPUT_NO_LORA"
  # echo "--------------------------------------------"

  # python inference/cli_demo.py \
  #   --prompt "$PROMPT" \
  #   --model_path "$MODEL_PATH" \
  #   --image_or_video_path "$IMAGE_PATH" \
  #   --generate_type "$GENERATE_TYPE" \
  #   --num_inference_steps "$NUM_STEPS" \
  #   --guidance_scale "$GUIDANCE_SCALE" \
  #   --num_frames "$NUM_FRAMES" \
  #   --fps "$FPS" \
  #   --output_path "$OUTPUT_NO_LORA"

  # 2) With LoRA
  OUTPUT_LORA="${OUTDIR}/${BASENAME}_withLORA_high.mp4"
  echo "=== Generating WITH LoRA ==="
  echo "Prompt: $PROMPT"
  echo "Image: $IMAGE_PATH"
  echo "Output: $OUTPUT_LORA"
  echo "--------------------------------------------"

  python inference/cli_demo.py \
    --prompt "$PROMPT" \
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
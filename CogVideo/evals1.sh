#!/usr/bin/env bash
#SBATCH -A berzelius-2025-23   # Your project/account
#SBATCH --gpus=1 -C "thin"     # GPU type needed
#SBATCH -t 1-00:00:00          # Time limit
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=128G
#SBATCH -J i2v_lora_eval
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

module load Mambaforge/23.3.1-1-hpc1-bdist
conda activate /proj/aicell/users/x_aleho/conda_envs/cogvideo

# --------------------------------------
# MODEL & LORA INFO
# --------------------------------------
MODEL_PATH="./models/CogVideoX1.5-5B-I2V"
LORA_PATH="./models/loras/evals_i2v/checkpoint-800"
GENERATE_TYPE="i2v"

# --------------------------------------
# INFERENCE PARAMS
# --------------------------------------
NUM_STEPS=50
GUIDANCE_SCALE=6.0
NUM_FRAMES=81
FPS=10

# --------------------------------------
# OUTPUT DIRECTORY
# --------------------------------------
OUTDIR="./test_generations/i2v_eval1_night"
mkdir -p "$OUTDIR"

# --------------------------------------
# IMAGE PATHS (2 starting frames)
# --------------------------------------

IMAGES=(
"/proj/aicell/users/x_aleho/video-diffusion/data/processed/idr0013/first_frames/LT0001_02-00223_01.png"
"/proj/aicell/users/x_aleho/video-diffusion/data/processed/idr0013/first_frames/LT0004_06-00058_01.png"
)

# --------------------------------------
# 3 PROMPTS, only differ in proliferation
# --------------------------------------
PROMPTS=(
"Time-lapse microscopy video with medium migration speed, high migration distance, and low proliferation."
"Time-lapse microscopy video with medium migration speed, high migration distance, and medium proliferation."
"Time-lapse microscopy video with medium migration speed, high migration distance, and high proliferation."
)
# For naming: we'll have a small array for the suffixes: [lowPROF, medPROF, highPROF].
PROLIF_SUFFIX=(
"lowPROF"
"medPROF"
"highPROF"
)

# --------------------------------------
# GENERATION LOOP
# For the *very first* image + first prompt => No LoRA & With LoRA
# For everything else => only With LoRA
# --------------------------------------
for i in "${!IMAGES[@]}"; do
  IMAGE_PATH="${IMAGES[$i]}"
  BASENAME=$(basename "$IMAGE_PATH" .png)

  for j in "${!PROMPTS[@]}"; do
    PROMPT_TEXT="${PROMPTS[$j]}"
    SUFFIX="${PROLIF_SUFFIX[$j]}"

    # If i=0 & j=0 => generate noLoRA & withLoRA
    if [[ "$i" -eq 0 && "$j" -eq 0 ]]; then
      OUTPUT_NO_LORA="${OUTDIR}/${BASENAME}_noLORA_${SUFFIX}.mp4"
      echo "=== Generating with NO LoRA ==="
      echo "Prompt: $PROMPT_TEXT"
      echo "Image: $IMAGE_PATH"
      echo "Output: $OUTPUT_NO_LORA"
      echo "--------------------------------------------"

      python inference/cli_demo.py \
        --prompt "$PROMPT_TEXT" \
        --model_path "$MODEL_PATH" \
        --image_or_video_path "$IMAGE_PATH" \
        --generate_type "$GENERATE_TYPE" \
        --num_inference_steps "$NUM_STEPS" \
        --guidance_scale "$GUIDANCE_SCALE" \
        --num_frames "$NUM_FRAMES" \
        --fps "$FPS" \
        --output_path "$OUTPUT_NO_LORA"
    fi

    # With LoRA
    OUTPUT_LORA="${OUTDIR}/${BASENAME}_withLORA_${SUFFIX}.mp4"
    echo "=== Generating WITH LoRA ==="
    echo "Prompt: $PROMPT_TEXT"
    echo "Image: $IMAGE_PATH"
    echo "Output: $OUTPUT_LORA"
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
      --output_path "$OUTPUT_LORA"

  done
done
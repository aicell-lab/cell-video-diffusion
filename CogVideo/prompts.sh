#!/usr/bin/env bash
#SBATCH -A berzelius-2025-23   # Your project/account
#SBATCH --gpus=1 -C "thin"     # GPU type needed
#SBATCH -t 1-00:00:00          # Time limit
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=128G
#SBATCH -J i2v_synonym_prompts
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

module load Mambaforge/23.3.1-1-hpc1-bdist
conda activate /proj/aicell/users/x_aleho/conda_envs/cogvideo

# --------------------------------------
# MODEL & LORA INFO
# --------------------------------------
MODEL_PATH="./models/CogVideoX1.5-5B-I2V"
LORA_PATH="./models/loras/evals_i2v/checkpoint-800"  # or whichever
GENERATE_TYPE="i2v"

# --------------------------------------
# INFERENCE PARAMS
# --------------------------------------
NUM_STEPS=50
GUIDANCE_SCALE=6.0
NUM_FRAMES=81
FPS=10

# (Optional) fix random seed for reproducibility
# SEED=42

# --------------------------------------
# OUTPUT DIRECTORY
# --------------------------------------
OUTDIR="./test_generations/i2v_synonym_spam"
mkdir -p "$OUTDIR"

# --------------------------------------
# SINGLE IMAGE PATH
# --------------------------------------
IMAGE_PATH="/proj/aicell/users/x_aleho/video-diffusion/data/processed/idr0013/first_frames/LT0001_02-00223_01.png"
BASENAME=$(basename "$IMAGE_PATH" .png)

# --------------------------------------
# PROMPT SYNONYMS
# --------------------------------------
SYNONYMS=(
"extremely high"
"massively high"
"super high"
"unbelievably high"
"wildly high"
"insanely high"
)

# We'll keep “medium migration speed” & “high migration distance” the same,
# only vary the proliferation part with synonyms above.

for syn in "${SYNONYMS[@]}"; do
  PROMPT="Time-lapse microscopy video with medium migration speed, high migration distance, and ${syn} proliferation."

  # Build a short suffix from the synonym to keep filename unique
  SAFE_SUFFIX=$(echo "$syn" | tr '[:space:]' '_' | tr -cd '[:alnum:]_')

  OUTPUT_PATH="${OUTDIR}/${BASENAME}_withLORA_${SAFE_SUFFIX}.mp4"

  echo "=================================================="
  echo "Prompt:  $PROMPT"
  echo "Image:   $IMAGE_PATH"
  echo "Steps:   $NUM_STEPS"
  echo "Scale:   $GUIDANCE_SCALE"
  echo "Frames:  $NUM_FRAMES"
  echo "FPS:     $FPS"
  echo "Output:  $OUTPUT_PATH"
  echo "=================================================="

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
    --output_path "$OUTPUT_PATH" \
    # --seed $SEED  # if you want a fixed seed

done

echo "All synonym-based runs finished!"
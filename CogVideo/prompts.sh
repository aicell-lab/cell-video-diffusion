#!/usr/bin/env bash
#SBATCH -A berzelius-2025-23   # Your project/account
#SBATCH --gpus=1 -C "thin"     # GPU type needed
#SBATCH -t 1-00:00:00          # Time limit
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=128G
#SBATCH -J i2v_prompts_multi_cfg
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
# We'll now use multiple guidance scales
GUIDANCE_SCALES=(10.0 15.0 20.0)
NUM_FRAMES=81
FPS=10

# (Optional) fix random seed for reproducibility
# SEED=42

# --------------------------------------
# OUTPUT DIRECTORY
# --------------------------------------
OUTDIR="./test_generations/i2v_prompts_extreme_multi_cfg"
mkdir -p "$OUTDIR"

# --------------------------------------
# SINGLE IMAGE PATH
# --------------------------------------
IMAGE_PATH="/proj/aicell/users/x_aleho/video-diffusion/data/processed/idr0013/first_frames/LT0001_02-00223_01.png"
BASENAME=$(basename "$IMAGE_PATH" .png)

# --------------------------------------
# FULL PROMPTS LIST
# --------------------------------------
OG_PROMPTS=(
"Time-lapse microscopy video with medium migration speed, high migration distance, and extremely high cell growth."
)

PROMPTS=(
  ""Time-lapse microscopy video with medium migration speed, high migration distance, and extremely high cell growth."",
  "A time-lapse microscopy video captures the extraordinary dynamics of cell migration and proliferation. The scene begins with a cluster of vibrant, fluorescently labeled cells at the center of the frame, set against a dark, contrast-enhanced background. These cells, exhibiting medium migration speed, steadily move outward in all directions, their paths forming intricate, branching patterns. As the video progresses, the cells cover a remarkable high migration distance, spreading uniformly across the field of view. Simultaneously, the rate of cell growth is astonishingly high, with individual cells rapidly dividing and multiplying, creating dense, interconnected networks. The visual spectacle of this accelerated cellular activity, highlighted by the glowing cell membranes and nuclei, underscores the remarkable resilience and adaptability of life at the microscopic level."
)

# Loop through each guidance scale
for GUIDANCE_SCALE in "${GUIDANCE_SCALES[@]}"; do
  for i in "${!PROMPTS[@]}"; do
    PROMPT="${PROMPTS[$i]}"
    OUTPUT_PATH="${OUTDIR}/${BASENAME}_withLORA_prompt${i}_cfg${GUIDANCE_SCALE}.mp4"

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
done

echo "All prompt-based runs with varying guidance scales finished!"
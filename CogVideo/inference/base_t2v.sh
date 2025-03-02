#!/usr/bin/env bash
#SBATCH -A berzelius-2025-23
#SBATCH --gpus=1 -C "thin"
#SBATCH -t 1-00:00:00
#SBATCH -J t2v_gen_baseline
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

module load Mambaforge/23.3.1-1-hpc1-bdist
conda activate /proj/aicell/users/x_aleho/conda_envs/cogvideo

# -------------------------------
# MODEL & LORA
# -------------------------------
MODEL_PATH="../models/CogVideoX1.5-5B"
GENERATE_TYPE="t2v"

PROMPTS=(
  "In this microscopy video, HeLa cells with ABCA1 knockdown appear less dynamic and show reduced movement compared to normal cells. The affected cells exhibit a flatter, more spread-out shape with less defined edges, indicating compromised structural integrity. Over time, these cells display minimal migration, barely shifting position across the field of view, creating a visual contrast with normal cells that actively move and change positions. As the video progresses, notice the lack of directional movement and clustering, where normal cells would typically form distinct pathways or trails. The overall scene presents a static, almost stagnant cellular environment, highlighting the impaired migration capability due to the ABCA1 knockdown."
  "In this video, HeLa cells with ABCA12 knockdown display an unusual increase in movement and speed, appearing more dynamic compared to normal cells. The cells are elongated and their edges are more irregular, showing a pronounced ability to migrate across the surface. Over time, you can observe these cells rapidly traversing the field of view, with their movements being more erratic and faster than their non-knockdown counterparts, which remain relatively stationary and rounded. The video highlights this contrast, as the affected cells seem to actively explore their environment, moving in various directions with noticeable changes in speed and direction."
  "In this video, HeLa cells with ABCA12 knockdown display an unusual increase in movement and speed, appearing more dynamic compared to normal cells. The cells are elongated and their edges are more irregular, showing a pronounced ability to migrate across the surface. Over time, you can observe these cells rapidly traversing the field of view, with their movements being more erratic and faster than their non-knockdown counterparts, which remain relatively stationary and rounded. The video highlights this contrast, as the affected cells seem to actively explore their environment, moving in various directions with noticeable changes in speed and direction."
)

# Use a single seed for all generations
SEED=9

# Output folder
OUTDIR="../../data/generated/test_generations_realval/t2v_baseline_visual_prompts"
mkdir -p "$OUTDIR"

# Fixed parameters for all generations
STEPS=50
SCALE=8
NUM_FRAMES=81
FPS=10

# Loop through each prompt
for i in "${!PROMPTS[@]}"; do
  PROMPT="${PROMPTS[$i]}"
  
  echo "=================================================="
  echo "Processing prompt $((i+1))/${#PROMPTS[@]}"
  echo "Prompt: $PROMPT"
  echo "Using seed: $SEED"
  
  # Construct output filename with simple naming
  OUTPUT_PATH="${OUTDIR}/prompt$((i+1))_seed${SEED}.mp4"
  
  echo "Output => ${OUTPUT_PATH}"
  
  python cli_demo.py \
    --prompt "$PROMPT" \
    --seed "$SEED" \
    --model_path "$MODEL_PATH" \
    --generate_type "$GENERATE_TYPE" \
    --num_inference_steps "$STEPS" \
    --guidance_scale "$SCALE" \
    --num_frames "$NUM_FRAMES" \
    --fps "$FPS" \
    --output_path "$OUTPUT_PATH"
done

echo "All done! Created ${#PROMPTS[@]} videos in $OUTDIR"
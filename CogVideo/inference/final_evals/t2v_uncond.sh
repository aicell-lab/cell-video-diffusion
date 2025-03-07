#!/usr/bin/env bash
#SBATCH -A berzelius-2025-23
#SBATCH --gpus=1
#SBATCH -t 1-00:00:00
#SBATCH -J t2v_uncond_50
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err


module load Mambaforge/23.3.1-1-hpc1-bdist
conda activate /proj/aicell/users/x_aleho/conda_envs/cogvideo

# -------------------------------
# MODEL SETUP
# -------------------------------
MODEL_PATH="../../models/CogVideoX1.5-5B"
SFT_PATH="../../models/sft/t2v-uncond/checkpoint-1200-fp32"

# -------------------------------
# GENERATION PARAMETERS
# -------------------------------
# Empty prompt for unconditional generation
PROMPT=""
# Fixed parameters
STEPS=50
SCALE=8
NUM_FRAMES=81
FPS=10

# Output folder
OUTDIR="../../../data/generated/final_evals/uncond_20videos"
mkdir -p "$OUTDIR"

# Generate 20 videos with different seeds
for SEED in {1..15}; do
  echo "=================================================="
  echo "Generating unconditional video $SEED of 50"
  
  # Construct output filename
  OUTPUT_PATH="${OUTDIR}/uncond_seed${SEED}.mp4"
  
  echo "Output => ${OUTPUT_PATH}"
  
  # Run generation
  python ../cli_demo.py \
    --prompt "$PROMPT" \
    --seed "$SEED" \
    --model_path "$MODEL_PATH" \
    --sft_path "$SFT_PATH" \
    --generate_type "t2v" \
    --num_inference_steps "$STEPS" \
    --guidance_scale "$SCALE" \
    --num_frames "$NUM_FRAMES" \
    --fps "$FPS" \
    --output_path "$OUTPUT_PATH"
done

echo "All done! Generated 50 unconditional videos in $OUTDIR"

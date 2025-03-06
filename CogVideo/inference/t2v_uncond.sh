#!/usr/bin/env bash
#SBATCH -A berzelius-2025-23
#SBATCH --gpus=1 -C "thin"
#SBATCH -t 1-00:00:00
#SBATCH -J t2v_phenotype
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# Get checkpoint from command line argument
CHECKPOINT=$1
echo "Using checkpoint: $CHECKPOINT"

module load Mambaforge/23.3.1-1-hpc1-bdist
conda activate /proj/aicell/users/x_aleho/conda_envs/cogvideo

# -------------------------------
# MODEL SETUP
# -------------------------------
MODEL_PATH="../models/CogVideoX1.5-5B"
SFT_PATH="../models/sft/IDR0013-FILTERED-uncond-t2v-uncond-final/checkpoint-${CHECKPOINT}-fp32"

# -------------------------------
# PHENOTYPE TESTING SETUP
# -------------------------------
# Base prompt without phenotype descriptions
BASE_PROMPT=""
# Seeds for variation
SEEDS=(9 21, 91)

# Fixed parameters
STEPS=30
SCALE=20
NUM_FRAMES=81
FPS=10

# Output folder
OUTDIR="../../data/generated/uncond_sft_${CHECKPOINT}"
mkdir -p "$OUTDIR"

# Loop through each phenotype variation
for SEED in "${SEEDS[@]}"; do
  # Create an identifier for this phenotype
  P_ID="seed${SEED}"
  
  # Generate variations with different seeds
  for SEED in "${SEEDS[@]}"; do
    echo "=================================================="
    echo "Processing with seed: $SEED"
    
    # Construct output filename
    OUTPUT_PATH="${OUTDIR}/${P_ID}_seed${SEED}.mp4"
    
    echo "Output => ${OUTPUT_PATH}"
    
    # Run generation with phenotypes
    python cli_demo.py \
      --prompt "$BASE_PROMPT" \
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
done

echo "All done! Generated phenotype variations in $OUTDIR"

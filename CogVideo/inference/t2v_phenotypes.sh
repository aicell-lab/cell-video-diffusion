#!/usr/bin/env bash
#SBATCH -A berzelius-2025-23
#SBATCH --gpus=1 -C "thin"
#SBATCH -t 1-00:00:00
#SBATCH -J t2v_phenotypes
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err


module load Mambaforge/23.3.1-1-hpc1-bdist
mamba activate /proj/aicell/users/x_aleho/conda_envs/cogvideo

# Modify these paths and parameters as needed:
MODEL_PATH="../models/CogVideoX1.5-5B"
SFT_PATH="../models/sft/IDR0013-FILTERED-2-t2v-special-single-final/checkpoint-1000-fp32"
PROMPT="Time lapse microscopy."
OUTPUT_DIR="../data/generated/t2v_phenotypes"
mkdir -p "$OUTPUT_DIR"

# You can adjust these as needed.
NUM_FRAMES=81
GUIDANCE_SCALE=6.0
NUM_INFERENCE_STEPS=30
SEED=42
FPS=10

# Define an array of phenotype combinations to test
declare -a phenotypes=(
    "0.2,0.8,0.0,1.0"
    "1.0,0.0,0.0,0.0"
    "0.0,0.0,0.0,0.0"
    "1.0,1.0,1.0,1.0"
    "0.5,0.5,0.5,0.5"
    "1.0,0.0,1.0,0.0"
)

# Iterate through each phenotype combination
for phenotype in "${phenotypes[@]}"; do
    echo "Processing phenotype combination: $phenotype"
    
    # Create a filename-safe version of the phenotype string
    safe_phenotype=$(echo $phenotype | tr ',' '-')
    OUTPUT_VIDEO="$OUTPUT_DIR/output_cat_phenotype_${safe_phenotype}.mp4"
    
    python cli_demo2.py \
        --model_path "$MODEL_PATH" \
        --sft_path "$SFT_PATH" \
        --prompt "$PROMPT" \
        --phenotypes "$phenotype" \
        --num_frames "$NUM_FRAMES" \
        --num_inference_steps "$NUM_INFERENCE_STEPS" \
        --guidance_scale "$GUIDANCE_SCALE" \
        --output_path "$OUTPUT_VIDEO" \
        --generate_type "t2v" \
        --fps "$FPS" \
        --seed "$SEED" \
        --dtype "bfloat16"
    
    echo "Finished generating video: $OUTPUT_VIDEO"
done

echo "All phenotype combinations processed!"

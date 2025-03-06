#!/usr/bin/env bash

module load Mambaforge/23.3.1-1-hpc1-bdist
mamba activate /proj/aicell/users/x_aleho/conda_envs/cogvideo

# Modify these paths and parameters as needed:
MODEL_PATH="../models/CogVideoX1.5-5B"
SFT_PATH="../models/sft/IDR0013-FILTERED-2-t2v-special-single-final/checkpoint-1000-fp32"
PROMPT="A cat climbing a tree"
PHENOTYPES="0.2,0.8,1.0,0.0"       # Example phenotypes, 4D
OUTPUT_DIR="../data/generated/t2v_phenotypes"
mkdir -p "$OUTPUT_DIR"

OUTPUT_VIDEO="$OUTPUT_DIR/output_cat_phenotypes.mp4"

# You can adjust these as needed.
NUM_FRAMES=81
GUIDANCE_SCALE=6.0
NUM_INFERENCE_STEPS=30
SEED=42
FPS=10

# Now call the new cli_demo.py
python cli_demo2.py \
    --model_path "$MODEL_PATH" \
    --sft_path "$SFT_PATH" \
    --prompt "$PROMPT" \
    --phenotypes "$PHENOTYPES" \
    --num_frames "$NUM_FRAMES" \
    --num_inference_steps "$NUM_INFERENCE_STEPS" \
    --guidance_scale "$GUIDANCE_SCALE" \
    --output_path "$OUTPUT_VIDEO" \
    --generate_type "t2v" \
    --fps "$FPS" \
    --seed "$SEED" \
    --dtype "bfloat16"

echo "Finished generating video: $OUTPUT_VIDEO"

#!/usr/bin/env bash

#############################################################################
# i2v_lora_test.sh
# Compares CogVideoX1.5-5B-I2V baseline vs. your new IDR0013 I2V LoRA on a few prompts.
#
# Prerequisites:
#   - "cli_demo.py" that can do i2v generation
#   - CogVideoX1.5-5B-I2V base model
#   - IDR0013 I2V LoRA checkpoint folder (with pytorch_lora_weights.safetensors)
#############################################################################

# Base model & LoRA
MODEL_PATH="./models/CogVideoX1.5-5B-I2V"
LORA_PATH="./models/loras/idr0013-i2v/checkpoint-25"  # e.g., pick a checkpoint folder
IMAGE_PATH="/proj/aicell/users/x_aleho/video-diffusion/data/processed/idr0013/first_frames/00265_01.png"
GENERATE_TYPE="i2v"

# Inference params
NUM_STEPS=30
GUIDANCE_SCALE=6.0
NUM_FRAMES=81  # 8N+1 for CogVideoX1.5
FPS=10
OUTDIR="./test_generations/idr0013_i2v_lora"
mkdir -p "$OUTDIR"

# Example prompts for cell imaging
PROMPTS=(
  "Time-lapse microscopy video of HeLa cells in LT0001_02_L1, with siRNA knockdown of PPM1F. Fluorescently labeled chromosomes are observed. Phenotype: no."
  "Time-lapse microscopy video of HeLa cells in LT0001_02_L3, with siRNA knockdown of CYP1A2. Fluorescently labeled chromosomes are observed. Phenotype: no."
  "Experiment with multi-lobed nuclei and incomplete cytokinesis over 5 time points."
)

for i in "${!PROMPTS[@]}"; do
  PROMPT="${PROMPTS[$i]}"
  SAFE_NAME=$(echo "$PROMPT" | sed 's/[^a-zA-Z0-9]/_/g' | cut -c1-40)

  ################################################################
  # 1) No LoRA
  ################################################################
  OUTPUT_NO_LORA="${OUTDIR}/prompt${i}_noLORA_${SAFE_NAME}.mp4"
  echo "=== Generating with NO LoRA ==="
  echo "Prompt: $PROMPT"
  echo "Output: $OUTPUT_NO_LORA"

  python inference/cli_demo.py \
    --prompt "$PROMPT" \
    --model_path "$MODEL_PATH" \
    --image_or_video_path "$IMAGE_PATH" \
    --generate_type "$GENERATE_TYPE" \
    --num_inference_steps "$NUM_STEPS" \
    --guidance_scale "$GUIDANCE_SCALE" \
    --num_frames "$NUM_FRAMES" \
    --fps "$FPS" \
    --output_path "$OUTPUT_NO_LORA"

  ################################################################
  # 2) With IDR0013 I2V LoRA
  ################################################################
  OUTPUT_LORA="${OUTDIR}/prompt${i}_IDRLOIRA_${SAFE_NAME}.mp4"
  echo "=== Generating WITH IDR0013 I2V LoRA ==="
  echo "Prompt: $PROMPT"
  echo "Output: $OUTPUT_LORA"

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

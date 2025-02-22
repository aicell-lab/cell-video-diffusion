#!/bin/bash
#SBATCH -A berzelius-2025-23     # Your project/account
#SBATCH --gpus=1 -C "thin"       # Number of GPUs needed
#SBATCH -t 1-00:00:00            # Time limit (e.g. 1 day)
#SBATCH --cpus-per-gpu=16        # CPU cores per GPU (adjust as needed)
#SBATCH --mem=128G               # Total memory (adjust as needed)
#SBATCH -J idr0013_lora_test_t2v # Job name
#SBATCH -o logs/%x_%j.out        # Standard output log
#SBATCH -e logs/%x_%j.err        # Standard error log

module load Mambaforge/23.3.1-1-hpc1-bdist

conda activate /proj/aicell/users/x_aleho/conda_envs/cogvideo

#############################################################################
# lora_test.sh
# Compares CogVideoX1.5-5B T2V baseline vs. new IDR0013 LoRA on cell prompts.
# 
# Prerequisites:
#   - "cli_demo.py" can generate T2V with CogVideoX (81 frames, 768×1360).
#   - The newly trained LoRA checkpoint is at, e.g., "./outputs/idr0013-lora/checkpoint-XYZ"
#############################################################################

MODEL_PATH="./models/CogVideoX1.5-5B"    # or your local copy of CogVideoX1.5-5B
LORA_PATH="./models/loras/idr0013-lora/checkpoint-125"   # adapt to your actual checkpoint folder
GENERATE_TYPE="t2v"
NUM_STEPS=30
GUIDANCE_SCALE=6.0
NUM_FRAMES=81   # 8N+1 for CogVideoX1.5, typically 81 or 161
FPS=8
OUTDIR="./idr_lora_test_videos"
mkdir -p "$OUTDIR"

# Example prompts:
# (1) A prompt from your training data or very similar
# (2) Another near-training prompt (mention the gene, phenotypes, etc.)
# (3) A brand-new "unseen" scenario to see how the LoRA influences it.

PROMPTS=(
  "Time-lapse microscopy video of HeLa cells in LT0002_24_F11, with siRNA knockdown of ALPL. Fluorescently labeled chromosomes are observed. Phenotype: no."
  "Time-lapse microscopy video of HeLa cells in LT0002_24_D1, with siRNA knockdown of STYX. Fluorescently labeled chromosomes are observed. Phenotype: no."
  "A purely hypothetical scenario: time-lapse of HeLa cells spontaneously forming giant multi-lobed nuclei under siRNA knockdown, with intense fluorescence."
)

for i in "${!PROMPTS[@]}"; do
  PROMPT="${PROMPTS[$i]}"
  SAFE_NAME=$(echo "$PROMPT" | sed 's/[^a-zA-Z0-9]/_/g' | cut -c1-40)

  #################################
  # 1) No LoRA
  #################################
  OUTPUT_NO_LORA="${OUTDIR}/prompt${i}_noLORA_${SAFE_NAME}.mp4"
  echo "=== Generating with NO LoRA ==="
  echo "Prompt: $PROMPT"
  echo "Output: $OUTPUT_NO_LORA"

  python inference/cli_demo.py \
    --prompt "$PROMPT" \
    --model_path "$MODEL_PATH" \
    --generate_type "$GENERATE_TYPE" \
    --num_inference_steps "$NUM_STEPS" \
    --guidance_scale "$GUIDANCE_SCALE" \
    --num_frames "$NUM_FRAMES" \
    --fps "$FPS" \
    --output_path "$OUTPUT_NO_LORA"

  #################################
  # 2) With IDR0013 LoRA
  #################################
  OUTPUT_LORA="${OUTDIR}/prompt${i}_IDRLORA_${SAFE_NAME}.mp4"
  echo "=== Generating WITH IDR0013 LoRA ==="
  echo "Prompt: $PROMPT"
  echo "Output: $OUTPUT_LORA"

  python inference/cli_demo.py \
    --prompt "$PROMPT" \
    --model_path "$MODEL_PATH" \
    --lora_path "$LORA_PATH" \
    --generate_type "$GENERATE_TYPE" \
    --num_inference_steps "$NUM_STEPS" \
    --guidance_scale "$GUIDANCE_SCALE" \
    --num_frames "$NUM_FRAMES" \
    --fps "$FPS" \
    --output_path "$OUTPUT_LORA"
done

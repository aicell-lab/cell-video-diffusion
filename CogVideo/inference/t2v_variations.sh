#!/usr/bin/env bash
#SBATCH -A berzelius-2025-23
#SBATCH --gpus=1 -C "thin"
#SBATCH -t 1-00:00:00
#SBATCH -J t2v_variations
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# Get checkpoint from command line argument
CHECKPOINT=$1
echo "Using checkpoint: $CHECKPOINT"

# Get type (lora or sft)
TYPE=$2
echo "Using model type: ${TYPE:-lora}"

module load Mambaforge/23.3.1-1-hpc1-bdist
conda activate /proj/aicell/users/x_aleho/conda_envs/cogvideo

# -------------------------------
# MODEL SETUP
# -------------------------------
MODEL_PATH="../models/CogVideoX1.5-5B"
GENERATE_TYPE="t2v"

if [[ "$TYPE" == "sft" ]]; then
  SFT_PATH="../models/sft/IDR0013-10plates-t2v-1/checkpoint-${CHECKPOINT}-fp32"
  MODEL_TYPE="sft"
else
  LORA_PATH="../models/loras/IDR0013-10plates-lora-t2v-r128/checkpoint-${CHECKPOINT}"
  MODEL_TYPE="lora"
fi

# -------------------------------
# STANDARD PROMPTS WITH MULTIPLE SEEDS
# -------------------------------
PROMPTS=(
  "<ALEXANDER> Time-lapse microscopy video with low proliferation."
  "<ALEXANDER> Time-lapse microscopy video with medium proliferation."
  "<ALEXANDER> Time-lapse microscopy video with high proliferation."
)

# Array of 10 seeds for variation
SEEDS=(9 21 42 123 456 789 1024 2048 3000 5555)

# Fixed parameters
STEPS=50
SCALE=8
NUM_FRAMES=81
FPS=10

# Output folder
OUTDIR="../../data/generated/test_generations/t2v_${MODEL_TYPE}_${CHECKPOINT}_variations"
mkdir -p "$OUTDIR"

# Loop through each prompt
for PROMPT in "${PROMPTS[@]}"; do
  # Create a simple identifier for this prompt
  if [[ "$PROMPT" == *"low proliferation"* ]]; then
    PROMPT_TYPE="pr-LOW"
  elif [[ "$PROMPT" == *"medium proliferation"* ]]; then
    PROMPT_TYPE="pr-MED"
  elif [[ "$PROMPT" == *"high proliferation"* ]]; then
    PROMPT_TYPE="pr-HIGH"
  else
    PROMPT_TYPE="pr-OTHER"
  fi
  
  # Generate 10 variations with different seeds
  for SEED in "${SEEDS[@]}"; do
    echo "=================================================="
    echo "Processing: $PROMPT with seed $SEED"
    echo "Using scale: $SCALE, steps: $STEPS, frames: $NUM_FRAMES"
    
    # Construct output filename
    OUTPUT_PATH="${OUTDIR}/t2v_${PROMPT_TYPE}_scale${SCALE}_seed${SEED}_S${STEPS}_F${NUM_FRAMES}.mp4"
    
    echo "Output => ${OUTPUT_PATH}"
    
    if [[ "$TYPE" == "sft" ]]; then
      # SFT model
      python cli_demo.py \
        --prompt "$PROMPT" \
        --seed "$SEED" \
        --model_path "$MODEL_PATH" \
        --sft_path "$SFT_PATH" \
        --generate_type "$GENERATE_TYPE" \
        --num_inference_steps "$STEPS" \
        --guidance_scale "$SCALE" \
        --num_frames "$NUM_FRAMES" \
        --fps "$FPS" \
        --output_path "$OUTPUT_PATH"
    else
      # LoRA model
      python cli_demo.py \
        --prompt "$PROMPT" \
        --seed "$SEED" \
        --model_path "$MODEL_PATH" \
        --lora_path "$LORA_PATH" \
        --generate_type "$GENERATE_TYPE" \
        --num_inference_steps "$STEPS" \
        --guidance_scale "$SCALE" \
        --num_frames "$NUM_FRAMES" \
        --fps "$FPS" \
        --output_path "$OUTPUT_PATH"
    fi
  done
done

echo "All done! Created ${#PROMPTS[@]}×${#SEEDS[@]} videos (3 prompts × 10 seeds) in $OUTDIR" 
#!/usr/bin/env bash
#SBATCH -A berzelius-2025-23
#SBATCH --gpus=1 -C "thin"
#SBATCH -t 1-00:00:00
#SBATCH -J t2v_extreme
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
# EXTREME PROMPTS
# -------------------------------
PROMPTS=(
  # Extreme proliferation levels
  "<ALEXANDER> Time-lapse microscopy video with extremely high proliferation."
  "<ALEXANDER> Time-lapse microscopy video with super high proliferation."
  "<ALEXANDER> Time-lapse microscopy video with maximum proliferation."
  "<ALEXANDER> Time-lapse microscopy video with explosive cell proliferation."
  
  # Cell behavior variations
  "<ALEXANDER> Time-lapse microscopy video with synchronized cell divisions."
  "<ALEXANDER> Time-lapse microscopy video with rapid cell migration."
  "<ALEXANDER> Time-lapse microscopy video with abnormal cell morphology."
  "<ALEXANDER> Time-lapse microscopy video with cells dividing unusually fast."
  
  # Cell conditions
  "<ALEXANDER> Time-lapse microscopy video of stressed cells."
  "<ALEXANDER> Time-lapse microscopy video of cells undergoing apoptosis."
  
  # Time variations
  "<ALEXANDER> Accelerated time-lapse microscopy of proliferating cells."
  "<ALEXANDER> Extended duration time-lapse of continuously dividing cells."
)

# Fixed parameters 
SEED=9
STEPS=50
SCALE=8
NUM_FRAMES=81
FPS=10

# Output folder
OUTDIR="../../data/generated/test_generations/t2v_${MODEL_TYPE}_${CHECKPOINT}_extreme"
mkdir -p "$OUTDIR"

# Loop through each prompt
for PROMPT in "${PROMPTS[@]}"; do
  # Create a simple identifier for this prompt based on key phrases
  if [[ "$PROMPT" == *"extremely high"* ]]; then
    PROMPT_TYPE="pr-EXTREME"
  elif [[ "$PROMPT" == *"super high"* ]]; then
    PROMPT_TYPE="pr-SUPER"
  elif [[ "$PROMPT" == *"maximum"* ]]; then
    PROMPT_TYPE="pr-MAX"
  elif [[ "$PROMPT" == *"explosive"* ]]; then
    PROMPT_TYPE="pr-EXPLOS"
  elif [[ "$PROMPT" == *"synchronized"* ]]; then
    PROMPT_TYPE="beh-SYNC"
  elif [[ "$PROMPT" == *"migration"* ]]; then
    PROMPT_TYPE="beh-MIGR"
  elif [[ "$PROMPT" == *"morphology"* ]]; then
    PROMPT_TYPE="beh-MORPH"
  elif [[ "$PROMPT" == *"unusually fast"* ]]; then
    PROMPT_TYPE="beh-FASTDIV"
  elif [[ "$PROMPT" == *"stressed"* ]]; then
    PROMPT_TYPE="cond-STRESS"
  elif [[ "$PROMPT" == *"apoptosis"* ]]; then
    PROMPT_TYPE="cond-APOP"
  elif [[ "$PROMPT" == *"Accelerated"* ]]; then
    PROMPT_TYPE="time-ACCEL"
  elif [[ "$PROMPT" == *"Extended"* ]]; then
    PROMPT_TYPE="time-EXTEND"
  else
    PROMPT_TYPE="OTHER"
  fi
  
  echo "=================================================="
  echo "Processing: $PROMPT"
  echo "Using scale: $SCALE, steps: $STEPS, frames: $NUM_FRAMES, seed: $SEED"
  
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

echo "All done! Created ${#PROMPTS[@]} extreme videos in $OUTDIR" 
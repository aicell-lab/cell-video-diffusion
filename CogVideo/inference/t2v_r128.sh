#!/usr/bin/env bash
#SBATCH -A berzelius-2025-23
#SBATCH --gpus=1 -C "thin"
#SBATCH -t 1-00:00:00
#SBATCH -J t2v_gen_r128
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# Get checkpoint from command line argument
CHECKPOINT=$1
echo "Using checkpoint: $CHECKPOINT"

# Get whether to vary seeds (optional)
VARY_SEED=$2
echo "Vary seeds: ${VARY_SEED:-false}"

# Get parameter to vary (optional)
VARY_PARAM=$3
echo "Parameter to vary: ${VARY_PARAM:-none}"

module load Mambaforge/23.3.1-1-hpc1-bdist
conda activate /proj/aicell/users/x_aleho/conda_envs/cogvideo

# -------------------------------
# MODEL & LORA
# -------------------------------
MODEL_PATH="../models/CogVideoX1.5-5B"
GENERATE_TYPE="t2v"
LORA_PATH="../models/loras/IDR0013-10plates-lora-t2v-r128/checkpoint-${CHECKPOINT}"

# -------------------------------
# EVALUATION SETUP: 3 PROMPTS, 3 SCALES
# -------------------------------
# One prompt for each proliferation rate
PROMPTS=(
  "<ALEXANDER> Time-lapse microscopy video with low proliferation."
  "<ALEXANDER> Time-lapse microscopy video with medium proliferation."
  "<ALEXANDER> Time-lapse microscopy video with high proliferation."
)

# Arrays of parameters that can be varied
SCALES=(8 15 20)
STEPS=(30 50 70)
FRAMES=(33 81 129)
SEEDS=(9 21)

# Fixed parameters for all generations
DEFAULT_SEED=9
DEFAULT_STEPS=50
DEFAULT_SCALE=8
DEFAULT_FRAMES=81
FPS=10

# Output folder
OUTDIR="../../data/generated/test_generations/t2v_r128_${CHECKPOINT}"
if [[ "$VARY_PARAM" == "scale" ]]; then
  OUTDIR="${OUTDIR}_vary_scale"
elif [[ "$VARY_PARAM" == "steps" ]]; then
  OUTDIR="${OUTDIR}_vary_steps"
elif [[ "$VARY_PARAM" == "frames" ]]; then
  OUTDIR="${OUTDIR}_vary_frames"
fi

if [[ "$VARY_SEED" == "true" ]]; then
  OUTDIR="${OUTDIR}_multi_seed"
fi

mkdir -p "$OUTDIR"

# Function to generate videos with optional seed variation
generate_video() {
    local prompt=$1
    local prompt_type=$2
    local scale=$3
    local steps=$4
    local frames=$5
    local default_seed=$6
    
    # Determine which seeds to use
    local seed_array=($default_seed)
    if [[ "$VARY_SEED" == "true" ]]; then
        seed_array=("${SEEDS[@]}")
    fi
    
    # Loop through seeds
    for seed in "${seed_array[@]}"; do
        echo "=================================================="
        echo "Processing: $prompt"
        echo "Using scale: $scale, steps: $steps, frames: $frames, seed: $seed"
        
        # Always include seed in the filename
        local output_path="${OUTDIR}/t2v_${prompt_type}_scale${scale}_seed${seed}_S${steps}_F${frames}.mp4"
        
        echo "Output => ${output_path}"
        
        python cli_demo.py \
          --prompt "$prompt" \
          --seed "$seed" \
          --model_path "$MODEL_PATH" \
          --lora_path "$LORA_PATH" \
          --generate_type "$GENERATE_TYPE" \
          --num_inference_steps "$steps" \
          --guidance_scale "$scale" \
          --num_frames "$frames" \
          --fps "$FPS" \
          --output_path "$output_path"
    done
}

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
  
  # Determine which parameter to vary
  if [[ "$VARY_PARAM" == "scale" ]]; then
    # Vary the scale parameter
    for SCALE in "${SCALES[@]}"; do
      generate_video "$PROMPT" "$PROMPT_TYPE" "$SCALE" "$DEFAULT_STEPS" "$DEFAULT_FRAMES" "$DEFAULT_SEED"
    done
  elif [[ "$VARY_PARAM" == "steps" ]]; then
    # Vary the steps parameter
    for STEPS_COUNT in "${STEPS[@]}"; do
      generate_video "$PROMPT" "$PROMPT_TYPE" "$DEFAULT_SCALE" "$STEPS_COUNT" "$DEFAULT_FRAMES" "$DEFAULT_SEED"
    done
  elif [[ "$VARY_PARAM" == "frames" ]]; then
    # Vary the number of frames
    for FRAME_COUNT in "${FRAMES[@]}"; do
      generate_video "$PROMPT" "$PROMPT_TYPE" "$DEFAULT_SCALE" "$DEFAULT_STEPS" "$FRAME_COUNT" "$DEFAULT_SEED"
    done
  else
    # Don't vary any parameter, just use default values
    generate_video "$PROMPT" "$PROMPT_TYPE" "$DEFAULT_SCALE" "$DEFAULT_STEPS" "$DEFAULT_FRAMES" "$DEFAULT_SEED"
  fi
done

# Calculate how many videos were generated
num_prompts=${#PROMPTS[@]}
num_videos=$num_prompts

if [[ "$VARY_PARAM" == "scale" ]]; then
  num_param_values=${#SCALES[@]}
  num_videos=$((num_prompts * num_param_values))
elif [[ "$VARY_PARAM" == "steps" ]]; then
  num_param_values=${#STEPS[@]}
  num_videos=$((num_prompts * num_param_values))
elif [[ "$VARY_PARAM" == "frames" ]]; then
  num_param_values=${#FRAMES[@]}
  num_videos=$((num_prompts * num_param_values))
fi

if [[ "$VARY_SEED" == "true" ]]; then
  num_videos=$((num_videos * ${#SEEDS[@]}))
  echo "All done! Created $num_videos videos with multiple seeds in $OUTDIR"
else
  echo "All done! Created $num_videos videos in $OUTDIR"
fi

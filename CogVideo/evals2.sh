#!/usr/bin/env bash
#SBATCH -A berzelius-2025-23
#SBATCH --gpus=1 -C "thin"
#SBATCH -t 1-00:00:00
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=128G
#SBATCH -J i2v_eval2_night
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

module load Mambaforge/23.3.1-1-hpc1-bdist
conda activate /proj/aicell/users/x_aleho/conda_envs/cogvideo

# -------------------------------
# MODEL & LORA
# -------------------------------
MODEL_PATH="./models/CogVideoX1.5-5B-I2V"
LORA_PATH="./models/loras/evals_i2v/checkpoint-800"
GENERATE_TYPE="i2v"

# -------------------------------
# ONE IMAGE & ONE PROMPT
# -------------------------------
IMAGE_PATH="/proj/aicell/users/x_aleho/video-diffusion/data/processed/idr0013/first_frames/LT0001_02-00223_01.png"
PROMPT="Time-lapse microscopy video with medium migration speed, high migration distance, and high proliferation."

# Output folder
OUTDIR="./test_generations/i2v_eval2_night"
mkdir -p "$OUTDIR"

# Basename for the image
BASENAME=$(basename "$IMAGE_PATH" .png)

# Parameter sets to test
STEPS_LIST=(30 50)
SCALE_LIST=(6 8)
FRAMES_LIST=(81 97)
FPS_LIST=(10 16)

# We'll do a 4-level nested loop: 2 × 2 × 2 × 2 = 16 combos
for steps in "${STEPS_LIST[@]}"; do
  for gs in "${SCALE_LIST[@]}"; do
    for nf in "${FRAMES_LIST[@]}"; do
      for fps_val in "${FPS_LIST[@]}"; do

        echo "=================================================="
        echo "Generating with LoRA: steps=${steps}, scale=${gs}, frames=${nf}, fps=${fps_val}"
        echo "Prompt: $PROMPT"
        echo "Image:  $IMAGE_PATH"
        
        # Construct output filename
        # e.g. LT0001_02-00223_01_S30_G6_F81_FPS10.mp4
        OUTPUT_PATH="${OUTDIR}/${BASENAME}_S${steps}_G${gs}_F${nf}_FPS${fps_val}.mp4"

        echo "Output => ${OUTPUT_PATH}"
        echo "--------------------------------------------------"

        python inference/cli_demo.py \
          --prompt "$PROMPT" \
          --model_path "$MODEL_PATH" \
          --lora_path "$LORA_PATH" \
          --image_or_video_path "$IMAGE_PATH" \
          --generate_type "$GENERATE_TYPE" \
          --num_inference_steps "$steps" \
          --guidance_scale "$gs" \
          --num_frames "$nf" \
          --fps "$fps_val" \
          --output_path "$OUTPUT_PATH"

      done
    done
  done
done

echo "All done! Created 16 videos in $OUTDIR"
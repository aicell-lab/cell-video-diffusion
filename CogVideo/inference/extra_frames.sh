#!/usr/bin/env bash
#SBATCH -A berzelius-2025-23
#SBATCH --gpus=1 -C "thin"
#SBATCH -t 1-00:00:00
#SBATCH --cpus-per-gpu=16
#SBATCH --mem=128G
#SBATCH -J i2v_extra_frames_r256
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

module load Mambaforge/23.3.1-1-hpc1-bdist
conda activate /proj/aicell/users/x_aleho/conda_envs/cogvideo

# -------------------------------
# MODEL & LORA
# -------------------------------
MODEL_PATH="../models/CogVideoX1.5-5B-I2V"
LORA_PATH="../models/loras/r256/checkpoint-450"
GENERATE_TYPE="i2v"

# --------------------------------
# PROMPT & IMAGES
# --------------------------------
PROMPT="Time-lapse microscopy video with medium migration speed, high migration distance, and high cell growth."

IMAGES=(
"/proj/aicell/users/x_aleho/video-diffusion/data/processed/idr0013/first_frames/LT0001_02-00223_01.png"
"/proj/aicell/users/x_aleho/video-diffusion/data/processed/idr0013/first_frames/LT0001_09-00374_01.png"
)

# --------------------------------
# Different Frame Counts
# --------------------------------
FRAMES_LIST=(129 113 97)

# --------------------------------
# Other Generation Params
# --------------------------------
STEPS=50
SCALE=8
FPS=10

# --------------------------------
# OUTPUT DIRECTORY
# --------------------------------
OUTDIR="../test_generations/extra_frames_r256_450"
mkdir -p "$OUTDIR"

# We'll run a double nested loop:
# For each of the 2 images => For each of the 3 frames => produce 1 video
for IMAGE_PATH in "${IMAGES[@]}"; do
    BASENAME=$(basename "$IMAGE_PATH" .png)

    for FRAMES in "${FRAMES_LIST[@]}"; do
        echo "==========================================="
        echo "Generating video with:"
        echo " Image:      $BASENAME"
        echo " Prompt:     $PROMPT"
        echo " Frames:     $FRAMES"
        echo " Steps:      $STEPS"
        echo " Scale:      $SCALE"
        echo " FPS:        $FPS"
        echo "-------------------------------------------"

        OUTPUT_NAME="${BASENAME}_S${STEPS}_G${SCALE}_F${FRAMES}_FPS${FPS}.mp4"
        OUTPUT_PATH="${OUTDIR}/${OUTPUT_NAME}"

        echo "Output => ${OUTPUT_PATH}"
        echo "-------------------------------------------"

        python cli_demo.py \
          --prompt "$PROMPT" \
          --model_path "$MODEL_PATH" \
          --lora_path "$LORA_PATH" \
          --image_or_video_path "$IMAGE_PATH" \
          --generate_type "$GENERATE_TYPE" \
          --num_inference_steps "$STEPS" \
          --guidance_scale "$SCALE" \
          --num_frames "$FRAMES" \
          --fps "$FPS" \
          --output_path "$OUTPUT_PATH"
    done
done

echo "All done! Generated 6 total videos in $OUTDIR"
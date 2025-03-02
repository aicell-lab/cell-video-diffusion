#!/usr/bin/env bash
#SBATCH -A berzelius-2025-23
#SBATCH --gpus=1 -C "thin"
#SBATCH -t 1-00:00:00
#SBATCH -J t2v_gen_baseline
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

module load Mambaforge/23.3.1-1-hpc1-bdist
conda activate /proj/aicell/users/x_aleho/conda_envs/cogvideo

# -------------------------------
# MODEL & LORA
# -------------------------------
MODEL_PATH="../models/CogVideoX1.5-5B"
GENERATE_TYPE="t2v"

PROMPTS=(
  "In this video, HeLa cells with ABCA6 knockdown are prominently active, displaying enhanced migration behavior. The cells appear elongated and dynamic, swiftly sliding across the field of view, contrasting with the typically more stationary and rounded shape of normal cells. Over time, these cells demonstrate increased distance covered and speed, moving in coordinated streams or individually, leaving trails of migration paths. The video captures their continual motion, with cells actively changing positions and forming new cellular configurations as they rapidly navigate the microscopic landscape. This heightened movement and altered cell morphology are key visual features distinguishing these knockdown cells from their normal counterparts.""
  "In this microscopy video, HeLa cells with ABCB5 knockdown exhibit increased motility, visibly traversing more significant distances at heightened speeds compared to normal cells. The affected cells display elongated and dynamic shapes, with their edges extending and retracting rapidly as they migrate across the field of view. Over time, observe a marked difference in movement patterns: the knockdown cells move in a more directed and purposeful manner, covering extensive areas, whereas the normal cells appear more static with occasional subtle shifts. The video should highlight the stark contrast between the two groups, with knockdown cells moving swiftly and continuously, creating a vivid display of cellular migration that stands out against the relatively stable backdrop of typical cells.""
  "HeLa cells with COPB1 knockdown exhibiting disrupted Golgi structure and eventual cell death. Initially, cells appear normal but gradually develop enlarged, vacuolated cytoplasm as the Golgi apparatus fragments. Over time, cells round up, membrane blebbing becomes visible, and cells detach from the surface before dying, while neighboring control cells maintain normal morphology and division patterns."
  "In the video, HeLa cells with ABCB9 knockdown appear less dynamic, exhibiting significantly reduced migration compared to normal cells. These cells are more rounded and less elongated, with a lack of directional movement across the frame. Initially, you might see some subtle twitching or slight movements, but as time progresses, their positions remain largely static, contrasting sharply with the surrounding active, migrating cells. Over time, the video reveals a stark difference: normal cells continue to extend, retract, and move, while the ABCB9 knockdown cells stay clustered and immobile, emphasizing their impaired ability to migrate. The video captures this striking visual contrast, highlighting the static nature of the affected cells amidst a backdrop of bustling cellular motion."
)

# Use a single seed for all generations
SEED=9

# Output folder
OUTDIR="../../data/generated/test_generations_realval/t2v_baseline_visual_prompts"
mkdir -p "$OUTDIR"

# Fixed parameters for all generations
STEPS=50
SCALE=8
NUM_FRAMES=81
FPS=10

# Loop through each prompt
for i in "${!PROMPTS[@]}"; do
  PROMPT="${PROMPTS[$i]}"
  
  echo "=================================================="
  echo "Processing prompt $((i+1))/${#PROMPTS[@]}"
  echo "Prompt: $PROMPT"
  echo "Using seed: $SEED"
  
  # Construct output filename with simple naming
  OUTPUT_PATH="${OUTDIR}/prompt$((i+1))_seed${SEED}.mp4"
  
  echo "Output => ${OUTPUT_PATH}"
  
  python cli_demo.py \
    --prompt "$PROMPT" \
    --seed "$SEED" \
    --model_path "$MODEL_PATH" \
    --generate_type "$GENERATE_TYPE" \
    --num_inference_steps "$STEPS" \
    --guidance_scale "$SCALE" \
    --num_frames "$NUM_FRAMES" \
    --fps "$FPS" \
    --output_path "$OUTPUT_PATH"
done

echo "All done! Created ${#PROMPTS[@]} videos in $OUTDIR"
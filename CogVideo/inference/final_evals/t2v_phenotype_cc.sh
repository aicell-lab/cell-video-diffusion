#!/usr/bin/env bash
#SBATCH -A berzelius-2025-23
#SBATCH --gpus=1 -C "thin"
#SBATCH -t 1-00:00:00
#SBATCH -J t2v_phenotype_cc_ARG1
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# Get HIGH or LOW from command line argument
CONDITION=$1
if [[ "$CONDITION" != "HIGH" && "$CONDITION" != "LOW" ]]; then
  echo "Error: First argument must be either HIGH or LOW"
  exit 1
fi

# Fixed number of videos to generate
NUM_VIDEOS=15
START_SEED=1
END_SEED=15

echo "Generating videos for cell count condition: $CONDITION"
echo "Generating $NUM_VIDEOS videos with seeds $START_SEED to $END_SEED"

module load Mambaforge/23.3.1-1-hpc1-bdist
conda activate /proj/aicell/users/x_aleho/conda_envs/cogvideo

# -------------------------------
# MODEL SETUP
# -------------------------------
MODEL_PATH="../../models/CogVideoX1.5-5B"
# Use phenotype-enabled model checkpoint
SFT_PATH="../../models/sft/t2v-phenotype/checkpoint-1200-fp32"

# -------------------------------
# GENERATION PARAMETERS
# -------------------------------
PROMPT="Time-lapse microscopy video of cells."
STEPS=50
SCALE=8
NUM_FRAMES=81
FPS=10

# -------------------------------
# PHENOTYPE SETUP FOR CELL COUNT
# -------------------------------
# Set phenotype values based on condition
# Format: "cell_count,proliferation,migration,cell_death"
if [[ "$CONDITION" == "HIGH" ]]; then
  # High cell count (1.0), other values default (0.5)
  PHENOTYPES="1.0,0.5,0.5,0.5"
else
  # Low cell count (0.0), other values default (0.5)
  PHENOTYPES="0.0,0.5,0.5,0.5"
fi

# -------------------------------
# GENERATE VIDEOS
# -------------------------------
# Create output directory
OUTDIR="../../../data/generated/final_evals/phenotype_cc"
mkdir -p "$OUTDIR"

# Process the specified condition
echo "=================================================="
echo "Processing phenotype: cc-$CONDITION"
echo "Phenotype values: $PHENOTYPES"

# Create subdirectory
PHENO_DIR="$OUTDIR/cc-$CONDITION"
mkdir -p "$PHENO_DIR"

# Generate videos with specified seed range
for SEED in $(seq $START_SEED $END_SEED); do
  echo "Generating video for seed $SEED (range $START_SEED-$END_SEED) for cc-$CONDITION"
  
  # Construct output filename
  OUTPUT_PATH="${PHENO_DIR}/cc-${CONDITION}_seed${SEED}.mp4"
  
  echo "Output => ${OUTPUT_PATH}"
  
  # Run generation using cli_demo2.py with phenotypes
  python ../cli_demo2.py \
    --prompt "$PROMPT" \
    --phenotypes "$PHENOTYPES" \
    --seed "$SEED" \
    --model_path "$MODEL_PATH" \
    --sft_path "$SFT_PATH" \
    --generate_type "t2v" \
    --num_inference_steps "$STEPS" \
    --guidance_scale "$SCALE" \
    --num_frames "$NUM_FRAMES" \
    --fps "$FPS" \
    --output_path "$OUTPUT_PATH"
done

echo "Completed generating videos with seeds $START_SEED-$END_SEED for cc-$CONDITION"
echo "All done! Generated $NUM_VIDEOS phenotype-conditioned videos for cc-$CONDITION in $PHENO_DIR" 
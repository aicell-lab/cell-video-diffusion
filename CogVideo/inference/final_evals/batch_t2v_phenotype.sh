#!/usr/bin/env bash
#SBATCH -A berzelius-2025-23
#SBATCH --gpus=1
#SBATCH -t 1-00:00:00
#SBATCH -J batch_t2v_phenotype_ARG1_ARG2
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

# Get phenotype type (pr, cc, ms) from first argument
PHENOTYPE=$1
if [[ "$PHENOTYPE" != "pr" && "$PHENOTYPE" != "cc" && "$PHENOTYPE" != "ms" ]]; then
  echo "Error: First argument must be one of: pr, cc, ms"
  echo "  pr = proliferation"
  echo "  cc = cell count"
  echo "  ms = migration speed"
  exit 1
fi

# Get HIGH or LOW from second argument
CONDITION=$2
if [[ "$CONDITION" != "HIGH" && "$CONDITION" != "LOW" ]]; then
  echo "Error: Second argument must be either HIGH or LOW"
  exit 1
fi

# Map phenotype code to display name
if [[ "$PHENOTYPE" == "pr" ]]; then
  PHENOTYPE_NAME="proliferation"
elif [[ "$PHENOTYPE" == "cc" ]]; then
  PHENOTYPE_NAME="cell count"
else
  PHENOTYPE_NAME="migration speed"
fi

echo "Generating videos for ${PHENOTYPE_NAME} condition: $CONDITION"
echo "Loading model once and generating 15 videos with seeds 1-15"

module load Mambaforge/23.3.1-1-hpc1-bdist
conda activate /proj/aicell/users/x_aleho/conda_envs/cogvideo

# Create the needed directory structure
mkdir -p configs

# Run the batch generator with the appropriate config
python batch_generator_phenotype.py --config configs/phenotype_${PHENOTYPE}_${CONDITION}.json 
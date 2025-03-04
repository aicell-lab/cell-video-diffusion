#!/usr/bin/env bash

# Create logs directory if it doesn't exist
mkdir -p logs

# Set the checkpoint to use
CHECKPOINT=850

# Submit job to vary just the seed
echo "Submitting job to vary only seeds..."
sbatch t2v_sft.sh $CHECKPOINT true
echo "Submitted job for checkpoint ${CHECKPOINT}, multiple seeds, no parameter variation"

# Submit job that varies scale with multiple seeds
echo "Submitting job to vary guidance scale with multiple seeds..."
sbatch t2v_sft.sh $CHECKPOINT true scale
echo "Submitted job for checkpoint ${CHECKPOINT}, multiple seeds, vary scale"

# Submit job that varies step count with single seed
echo "Submitting job to vary inference steps with single seed..."
sbatch t2v_sft.sh $CHECKPOINT false steps
echo "Submitted job for checkpoint ${CHECKPOINT}, single seed, vary steps"

# Submit job that varies frame count with multiple seeds
echo "Submitting job to vary frame count with multiple seeds..."
sbatch t2v_sft.sh $CHECKPOINT true frames
echo "Submitted job for checkpoint ${CHECKPOINT}, multiple seeds, vary frames"

echo "All evaluation jobs have been submitted!" 
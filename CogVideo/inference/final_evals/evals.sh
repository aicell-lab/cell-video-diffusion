#!/usr/bin/env bash

# Create logs directory if it doesn't exist
mkdir -p logs

echo "==================================================="
echo "Starting evaluation job submissions"
echo "==================================================="

# -------------------------
# PHENOTYPE JOBS - CELL COUNT
# -------------------------
echo "Submitting cell count HIGH condition (phenotype)"
sbatch t2v_phenotype_cc.sh HIGH
echo "Waiting 5 seconds before next submission..."
sleep 5

echo "Submitting cell count LOW condition (phenotype)"
sbatch t2v_phenotype_cc.sh LOW
echo "Waiting 5 seconds before next submission..."
sleep 5

# -------------------------
# PHENOTYPE JOBS - PROLIFERATION
# -------------------------
echo "Submitting proliferation HIGH condition (phenotype)"
sbatch t2v_phenotype_pr.sh HIGH
echo "Waiting 5 seconds before next submission..."
sleep 5

echo "Submitting proliferation LOW condition (phenotype)"
sbatch t2v_phenotype_pr.sh LOW
echo "Waiting 5 seconds before next submission..."
sleep 5

# -------------------------
# UNCONDITIONAL GENERATION
# -------------------------
echo "Submitting unconditional generation job (81 frames)"
sbatch t2v_uncond.sh
echo "Waiting 5 seconds before next submission..."
sleep 5

echo "Submitting unconditional generation job (129 frames)"
sbatch t2v_uncond_frames.sh
echo "Waiting 5 seconds before next submission..."
sleep 5


# -------------------------
# PHENOTYPE JOBS - MIGRATION SPEED
# -------------------------
echo "Submitting migration speed HIGH condition (phenotype)"
sbatch t2v_phenotype_ms.sh HIGH
echo "Waiting 5 seconds before next submission..."
sleep 5

echo "Submitting migration speed LOW condition (phenotype)"
sbatch t2v_phenotype_ms.sh LOW
echo "Waiting 5 seconds before next submission..."
sleep 5


# -------------------------
# TEXT PROMPT JOBS - PROLIFERATION
# -------------------------
echo "Submitting proliferation HIGH condition (text prompt)"
sbatch t2v_prompt_pr.sh HIGH
echo "Waiting 5 seconds before next submission..."
sleep 5

echo "Submitting proliferation LOW condition (text prompt)"
sbatch t2v_prompt_pr.sh LOW
echo "Waiting 5 seconds before next submission..."
sleep 5

# -------------------------
# TEXT PROMPT JOBS - CELL COUNT
# -------------------------
echo "Submitting cell count HIGH condition (text prompt)"
sbatch t2v_prompt_cc.sh HIGH
echo "Waiting 5 seconds before next submission..."
sleep 5

echo "Submitting cell count LOW condition (text prompt)"
sbatch t2v_prompt_cc.sh LOW
echo "Waiting 5 seconds before next submission..."
sleep 5

# -------------------------
# TEXT PROMPT JOBS - MIGRATION SPEED
# -------------------------
echo "Submitting migration speed HIGH condition (text prompt)"
sbatch t2v_prompt_ms.sh HIGH
echo "Waiting 5 seconds before next submission..."
sleep 5

echo "Submitting migration speed LOW condition (text prompt)"
sbatch t2v_prompt_ms.sh LOW
echo "Waiting 5 seconds before next submission..."
sleep 5

# -------------------------
# SUMMARY
# -------------------------
echo "==================================================="
echo "All evaluation jobs submitted!"
echo "==================================================="
echo "Summary of submitted jobs:"
echo "- 1 job for unconditional generation (81 frames, 20 videos)"
echo "- 1 job for unconditional generation (129 frames, 20 videos)"
echo "- 1 job for proliferation HIGH (text prompt, 15 videos)"
echo "- 1 job for proliferation LOW (text prompt, 15 videos)"
echo "- 1 job for cell count HIGH (text prompt, 15 videos)"
echo "- 1 job for cell count LOW (text prompt, 15 videos)"
echo "- 1 job for migration speed HIGH (text prompt, 15 videos)"
echo "- 1 job for migration speed LOW (text prompt, 15 videos)"
echo "- 1 job for cell count HIGH (phenotype, 15 videos)"
echo "- 1 job for cell count LOW (phenotype, 15 videos)"
echo "- 1 job for proliferation HIGH (phenotype, 15 videos)"
echo "- 1 job for proliferation LOW (phenotype, 15 videos)"
echo "- 1 job for migration speed HIGH (phenotype, 15 videos)"
echo "- 1 job for migration speed LOW (phenotype, 15 videos)"
echo "==================================================="
echo "Total: 14 jobs submitted, generating 220 videos"
echo "==================================================="
echo "Check job status with: squeue -u \$USER"
echo "==================================================="

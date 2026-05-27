#!/bin/bash
#SBATCH --job-name=prob_nn     # Job name
#SBATCH --output=result/slurm_%j.log        # Console output goes here (%j = job ID)
#SBATCH --partition=ComputeQ           # Use DebugQ for testing, ComputeQ for long runs
#SBATCH --time=15:00:00              # Max time (HH:MM:SS) - Adjust as needed
#SBATCH --gres=gpu:1                 # Request 1 GPU (SLURM will pick GPU 5, 6, or 7)
#SBATCH --cpus-per-task=8            # CPU cores per GPU
#SBATCH --mem=32G                    # System RAM

# --- Environment Setup (UV) ---
# Activate the UV virtual environment located in your project folder
source .venv/bin/activate

# --- Run Your Code ---
# The -u flag forces unbuffered output, so logs appear immediately in the slurm log
echo "Starting job on node: $(hostname)"
echo "GPU Allocated: $CUDA_VISIBLE_DEVICES"

python -u src/main.py  # <--- REPLACE 'main.py' with your actual script name
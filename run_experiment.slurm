#!/bin/bash
#SBATCH --job-name=dripps
#SBATCH --account=eehpc-dev-2026d06-183g
#SBATCH --partition=normal-a100-40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# One GPU experiment per job on Deucalion (EuroHPC project EHPC-DEV-2026D06-183).
# Submit from the repo root; everything after the script name is the experiment command:
#   sbatch run_experiment.slurm experiments/10_decision.py --model qwen3.5-9b
#   sbatch -J enc_bertimbau run_experiment.slurm experiments/08_encoder.py --variant both --checkpoint bertimbau
#   sbatch --time=12:00:00 run_experiment.slurm experiments/10_decision.py --model qwen3.8-27b-fp8
# Deucalion rules (docs.macc.fccn.pt/jobs/gpu): GPU accounts end in "g"; 32 CPUs per GPU
# (asking for more is billed as extra GPUs). Compute nodes are offline: cache models on
# the login node first, e.g.  HF_HOME=../hf_cache hf download Qwen/Qwen3.5-9B
# The "logs/" dir must exist before sbatch (Slurm opens --output before the script runs).
set -euo pipefail
[[ $# -gt 0 ]] || { echo "usage: sbatch run_experiment.slurm <script.py> [args...]"; exit 2; }

cd "${SLURM_SUBMIT_DIR}"
ml Python/3.12.3-GCCcore-13.3.0 CUDA/12.4.0
source .venv/bin/activate   # same venv the Makefile uses: python -m venv .venv && pip install -r requirements-neural.txt

export HF_HOME="${HF_HOME:-$(dirname "${SLURM_SUBMIT_DIR}")/hf_cache}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONNOUSERSITE=1   # ~/.local packages must not shadow the venv
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1   # live logs

echo "job ${SLURM_JOB_ID} on ${SLURM_NODELIST}, $(date)"; nvidia-smi -L
python "$@"
echo "done $(date)"

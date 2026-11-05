#!/bin/bash
#SBATCH --job-name=openjev
#SBATCH --account=eehpc-dev-2026d06-183g
#SBATCH --partition=dev-a100-80
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# T5 through OpenJev's decision server (docs/OPENJEV_PLAN.md): vLLM serves the weights, OpenJev's helper/shim.py
# exposes POST /v1/systemone on top, and 10_decision.py scores every text through it (--engine http).
# Submit from the repo root; extra arguments go to 10_decision.py (e.g. --smoke):
#   sbatch -J oj_smoke --export=ALL,MODEL=../models/openjev-FP8,NAME=openjev-27b run_openjev.slurm --smoke
#   sbatch -J oj_full  --export=ALL,MODEL=../models/openjev-FP8,NAME=openjev-27b run_openjev.slurm
#   sbatch -J q38_shim --export=ALL,MODEL=Qwen/Qwen3.8-27B-FP8,NAME=qwen3.8-27b-shim run_openjev.slurm
# Needs (login node, once): ../models/openjev-FP8, ../models/openjev-helper/helper/shim.py, .venv-openjev with
# vllm==0.29.0 openai==3.16.2 httpx==0.28.1, and the main .venv. Compute nodes are offline.
set -euo pipefail
MODEL="${MODEL:?set MODEL=<local dir or cached Hub id>}"
NAME="${NAME:?set NAME=<cache name>}"
READOUT_T="${READOUT_T:-0.85}"  # OpenJev's quick-start readout temperature; recorded in meta.json via /v1/version
SERVER_TIMEOUT="${SERVER_TIMEOUT:-1200}"

cd "${SLURM_SUBMIT_DIR}"
ml Python/3.12.3-GCCcore-13.3.0 CUDA/12.4.0
export HF_HOME="$(dirname "${SLURM_SUBMIT_DIR}")/hf_cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
# Jobs packed on one node share its loopback: derive the ports from the job id so they never collide.
P=$(( 20000 + ${SLURM_JOB_ID} % 20000 * 2 ))
SHIM_PORT=$(( P + 1 ))
# A Hub id is resolved from the offline cache to a local snapshot, so the shim's tokenizer loads offline too.
if [[ -d "${MODEL}" ]]; then MODEL_DIR="$(cd "${MODEL}" && pwd)"; else
  MODEL_DIR="$(.venv/bin/python -c "from huggingface_hub import snapshot_download as s; print(s('${MODEL}', local_files_only=True))")"; fi

echo "job ${SLURM_JOB_ID} on ${SLURM_NODELIST}, $(date)"; nvidia-smi -L
echo "model ${MODEL} -> ${MODEL_DIR}; name ${NAME}; READOUT_T ${READOUT_T}; vLLM :${P}, shim :${SHIM_PORT}"

PIDS=()
cleanup() { for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null || true; done; wait 2>/dev/null || true; }
trap cleanup EXIT

wait_for() {  # url, pid, log
  for ((t = 0; t < SERVER_TIMEOUT; t += 10)); do
    kill -0 "$2" 2>/dev/null || { echo "ERROR: server exited early; tail of $3:"; tail -n 50 "$3"; exit 1; }
    curl -sf "$1" >/dev/null && { echo "ready after ~${t}s: $1"; return; }
    sleep 10
  done
  echo "ERROR: $1 not ready after ${SERVER_TIMEOUT}s; tail of $3:"; tail -n 50 "$3"; exit 1
}

VLLM_LOG="logs/${NAME}_vllm_${SLURM_JOB_ID}.log"
.venv-openjev/bin/vllm serve "${MODEL_DIR}" --host 127.0.0.1 --port "${P}" --served-model-name qwen \
  --max-model-len 16384 --gpu-memory-utilization 0.90 --trust-remote-code --max-num-seqs 16 --max-logprobs 64 \
  --gdn-prefill-backend triton --quantization fp8 --enable-prefix-caching --seed 0 > "${VLLM_LOG}" 2>&1 &
PIDS+=($!); wait_for "http://127.0.0.1:${P}/v1/models" "${PIDS[-1]}" "${VLLM_LOG}"

SHIM_LOG="logs/${NAME}_shim_${SLURM_JOB_ID}.log"
VLLM="http://127.0.0.1:${P}/v1" TOKENIZER="${MODEL_DIR}" SHIM_MODEL="${MODEL_DIR}" READOUT_T="${READOUT_T}" \
  .venv-openjev/bin/python ../models/openjev-helper/helper/shim.py --host 127.0.0.1 --port "${SHIM_PORT}" > "${SHIM_LOG}" 2>&1 &
PIDS+=($!); wait_for "http://127.0.0.1:${SHIM_PORT}/v1/version" "${PIDS[-1]}" "${SHIM_LOG}"
curl -s "http://127.0.0.1:${SHIM_PORT}/v1/version"; echo

source .venv/bin/activate
python experiments/10_decision.py --engine http --url "http://127.0.0.1:${SHIM_PORT}/v1/systemone" \
  --name "${NAME}" --id qwen "$@"
echo "done $(date)"

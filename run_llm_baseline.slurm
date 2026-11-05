#!/bin/bash
#SBATCH --job-name=claimpt_llm_baseline
#SBATCH --account=f202600030aivlabdeucaliong
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --partition=normal-a100-40
#SBATCH --output=logs/claimpt_llm_baseline_%j.out
#SBATCH --error=logs/claimpt_llm_baseline_%j.err

# ── ClaimPT — local open-weights LLM claim-extraction baseline ────────────────
# ONE parametrized template that serves any registered model locally with vLLM
# (offline, from the HF cache) and runs the SAME langextract few-shot pipeline
# used by the AMALIA / Gemini baselines, then validates the output.
#
# Per-model config (HF id, served name, context budgets, vLLM extras) lives in
# scripts/llm_baselines/registry.py — the single source of truth. This script
# just `eval`s that spec. Launch all models in parallel with submit_all_baselines.sh.
#
# Pre-requisite (ONCE, on the login node with internet):
#   export HF_TOKEN=<token>        # for gated models (llama3.1-8b, gemma2-9b)
#   python download_all_models.py  # caches every baseline into <repo_parent>/hf_cache
#
# Usage:
#   sbatch -J claimpt_qwen2.5-7b --export=ALL,MODEL=qwen2.5-7b slurm/run_llm_baseline.slurm
#
# Required:
#   MODEL                : registry key (amalia, llama3.1-8b, qwen2.5-7b, gemma2-9b,
#                          mistral-7b, eurollm-9b, salamandra-7b, gervasio-8b)
# Optional overrides (via --export=ALL,VAR=value):
#   PORT                 : local vLLM port (default: 8000)
#   LIMIT                : process only first N test articles (default: all)
#   SERVER_TIMEOUT       : seconds to wait for vLLM readiness (default: 1200)
#   FORCE_RERUN          : "1" to re-extract even if the output already exists
#   MIN_PARSE_SUCCESS    : sanity-gate floor on parse-success rate (default: 0.80)
#   MAX_MODEL_LEN_OVERRIDE / GPU_MEM_UTIL_OVERRIDE : tweak vLLM without editing the registry
# ─────────────────────────────────────────────────────────────────────────────
set -o pipefail

MODEL="${MODEL:?ERROR: set MODEL=<registry key>, e.g. --export=ALL,MODEL=qwen2.5-7b}"
# UNIQUE per-job port. CRITICAL: SLURM packs several single-GPU jobs onto ONE node
# (each gets its own GPU, but they SHARE the host's loopback). With a hardcoded
# 8000 their vLLM servers contend for the same port and every job's langextract
# client hits a SIBLING server → "404 The model `<X>` does not exist", silently
# poisoning results (the model itself is fine). Derive a unique, deterministic
# port from the job id so co-located jobs never collide. Override with PORT=.
if [[ -z "${PORT:-}" ]]; then
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
        PORT=$(( 20000 + SLURM_JOB_ID % 40000 ))
    else
        PORT=8000
    fi
fi
LIMIT="${LIMIT:-}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-1200}"
FORCE_RERUN="${FORCE_RERUN:-0}"
MIN_PARSE_SUCCESS="${MIN_PARSE_SUCCESS:-0.80}"

# SLURM_SUBMIT_DIR = the ClaimPT repo root (where sbatch was called).
# SLURM scripts live in <repo>/slurm/. Resolve REPO_DIR to the repo root whether
# sbatch was called from the repo root (`sbatch slurm/<script>`) or from slurm/.
_SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ "$(basename "${_SUBMIT_DIR}")" == "slurm" ]]; then
    REPO_DIR="$(dirname "${_SUBMIT_DIR}")"
else
    REPO_DIR="${_SUBMIT_DIR}"
fi
BASE_DIR="$(dirname "${REPO_DIR}")"
VENV_DIR="${BASE_DIR}/myenv311"
HF_CACHE_DIR="${BASE_DIR}/hf_cache"

cd "${REPO_DIR}"

ml Python/3.12.3-GCCcore-13.3.0 OpenMPI/5.0.3-GCC-13.3.0 CUDA/12.4.0 NCCL/2.20.5-GCCcore-13.3.0-CUDA-12.4.0

source "${VENV_DIR}/bin/activate"

# Offline: compute nodes have no network. Models must be pre-cached by
# download_all_models.py on the login node.
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_HOME="${HF_CACHE_DIR}"
export HF_HUB_CACHE="${HF_CACHE_DIR}/hub"
export TRANSFORMERS_CACHE="${HF_CACHE_DIR}/hub"

# Isolate from ~/.local (user-site): a stray package there must never shadow the
# venv. A user-site transformers silently kept a v5 in play and broke vLLM's
# tokenizer cache even after the venv was "fixed". This forces venv-only imports.
export PYTHONNOUSERSITE=1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="${REPO_DIR}:${PYTHONPATH:-}"

mkdir -p logs results/llm_baselines

# ── Resolve the per-model spec from the registry (single source of truth) ─────
SPEC="$(python -m scripts.llm_baselines.registry "${MODEL}")" || {
    echo "ERROR: unknown MODEL='${MODEL}'. Known keys:"
    python -c "from scripts.llm_baselines.registry import all_keys; print('  ' + ', '.join(all_keys()))"
    exit 2
}
eval "${SPEC}"

# Optional vLLM overrides (tweak without editing the registry).
MAX_MODEL_LEN="${MAX_MODEL_LEN_OVERRIDE:-${MAX_MODEL_LEN}}"
GPU_MEM_UTIL="${GPU_MEM_UTIL_OVERRIDE:-${GPU_MEM_UTIL}}"

OUTPUT="results/llm_baselines/${MODEL}_langextract.jsonl"
REPORT="results/llm_baselines/${MODEL}_report.txt"
BASE_URL="http://localhost:${PORT}"
SERVER_LOG="logs/${MODEL}_vllm_${SLURM_JOB_ID:-local}.log"

echo "================================================================================"
echo "ClaimPT — local LLM claim-extraction baseline"
echo "================================================================================"
echo "Job ID:          ${SLURM_JOB_ID:-<none>}"
echo "Node:            ${SLURM_NODELIST:-<none>}"
echo "Model key:       ${MODEL}"
echo "HF model:        ${HF_MODEL_ID} (served as '${SERVED_NAME}')"
echo "Max model len:   ${MAX_MODEL_LEN}   GPU mem util: ${GPU_MEM_UTIL}   TP: ${TP_SIZE}"
echo "Budgets:         few_shot_chars=${FEW_SHOT_MAX_CHARS} char_buf=${MAX_CHAR_BUFFER} out_tok=${MAX_OUTPUT_TOKENS}"
echo "Extra vLLM args: ${EXTRA_VLLM_ARGS:-<none>}"
echo "Base URL:        ${BASE_URL}"
echo "Output:          ${OUTPUT}"
echo "Limit:           ${LIMIT:-<all>}"
echo "Server log:      ${SERVER_LOG}"
echo "Started:         $(date)"
echo "================================================================================"

# ── SKIP_IF_DONE: don't waste a GPU allocation re-running a finished model ────
if [[ "${FORCE_RERUN}" != "1" && -s "${OUTPUT}" ]]; then
    echo "[SKIP] ${OUTPUT} already exists. Set FORCE_RERUN=1 to re-extract."
    exit 0
fi

# ── Start local vLLM server ──────────────────────────────────────────────────
echo "Launching vLLM server..."
vllm serve "${HF_MODEL_ID}" \
    --served-model-name "${SERVED_NAME}" \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --dtype "${DTYPE}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --seed 0 \
    ${EXTRA_VLLM_ARGS} \
    > "${SERVER_LOG}" 2>&1 &
SERVER_PID=$!

# Always tear the server down on exit (success, failure, or cancellation).
cleanup() {
    echo "Shutting down vLLM server (PID ${SERVER_PID})..."
    kill "${SERVER_PID}" 2>/dev/null
    wait "${SERVER_PID}" 2>/dev/null
}
trap cleanup EXIT

# ── Wait for readiness ───────────────────────────────────────────────────────
echo "Waiting up to ${SERVER_TIMEOUT}s for the server to become ready..."
ready=0
for ((t=0; t<SERVER_TIMEOUT; t+=10)); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "ERROR: vLLM server process exited prematurely. Tail of ${SERVER_LOG}:"
        tail -n 50 "${SERVER_LOG}"
        exit 1
    fi
    if curl -s "${BASE_URL}/v1/models" | grep -q "${SERVED_NAME}"; then
        ready=1
        echo "Server is ready after ~${t}s."
        break
    fi
    sleep 10
done
if [[ "${ready}" -ne 1 ]]; then
    echo "ERROR: server did not become ready within ${SERVER_TIMEOUT}s. Tail of ${SERVER_LOG}:"
    tail -n 50 "${SERVER_LOG}"
    exit 1
fi

# ── Extraction (shared langextract pipeline, pointed at localhost) ────────────
LIMIT_FLAG=()
[[ -n "${LIMIT}" ]] && LIMIT_FLAG=(--limit "${LIMIT}")

run_extract() {
    python scripts/04_run_langextract.py \
        --provider local \
        --base-url "${BASE_URL}" \
        --model "${SERVED_NAME}" \
        --max-char-buffer "${MAX_CHAR_BUFFER}" \
        --max-output-tokens "${MAX_OUTPUT_TOKENS}" \
        --few-shot-example-max-chars "${FEW_SHOT_MAX_CHARS}" \
        --output-file "${OUTPUT}" \
        "$@"
}

echo ""
echo "### Extraction — ${MODEL} ($(date))"
if ! run_extract "${LIMIT_FLAG[@]}"; then
    echo "ERROR: extraction failed for ${MODEL}."
    exit 1
fi

# Auto-retry pass: heal any documents whose segments ALL failed during the main
# run (transient blips), while the server is still up.
echo ""
echo "### Auto-retry pass — ${MODEL} ($(date))"
run_extract --retry-empty "${OUTPUT}" "${LIMIT_FLAG[@]}" || \
    echo "[warn] retry-empty pass returned non-zero (continuing to sanity gate)"

# ── Sanity gate: fail loudly if the output is unusable ───────────────────────
echo ""
echo "### Sanity gate — ${MODEL} ($(date))"
python scripts/llm_baselines/sanity_check.py "${OUTPUT}" \
    --min-parse-success "${MIN_PARSE_SUCCESS}" \
    --report-file "${REPORT}" \
    "${LIMIT_FLAG[@]}"
gate=$?

echo ""
echo "================================================================================"
echo "Finished: $(date)   sanity-gate exit=${gate}"
echo "Output:   ${OUTPUT}"
echo "Report:   ${REPORT}"
echo "Evaluate all baselines together once every job is done:"
echo "  python scripts/05_evaluate.py --compare-all results/llm_baselines/ --output-csv comparison_strict.csv"
echo "  python scripts/05_evaluate.py --compare-all results/llm_baselines/ --relaxed --output-csv comparison_relaxed.csv"
echo "================================================================================"
exit ${gate}

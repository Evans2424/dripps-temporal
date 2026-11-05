# OpenJev experiment: does decision tuning change what a model knows, or only how it answers?

**Status:** plan, not run. Written 2026-11-05. **Freeze §2 (commit hash below) before the first full run.**
Frozen at commit: `__________` · date: `__________`

## 1. Why this experiment

**The T5 results so far suggest a hypothesis.** Decision training improves a model's *raw* answers, not its knowledge.
- Jev beats the stock Qwen3.8-27B raw (0.628 against 0.560).
- Once both are calibrated they are level (0.632 against 0.668).
- Both add the same +0.043 to B3.

Jev can't test this, because it is closed and its base model is unknown.

**OpenJev can.** It is an open decision model whose `NOTICE` states its base: **`Qwen/Qwen3.8-27B`** (revision `1d4bf0f`),
the model already run as the T5 headline (FP8 checkpoint). It is tuned "so that a single position carries the
decision", to pick the correct option and to stay consistent when the options are reordered.

**Facts from its cards and `helper/shim.py`:**
- Weights CC BY-NC 4.0 (research use is fine); the helper is Apache 2.0.
- 27B FP8 is about 29 GB. There is also a Flash 9B (base not checked yet).
- It is served by `vllm serve`, with `helper/shim.py` exposing `POST /v1/systemone` in the request format of TypeSafe's Jev.
- The shim assigns the letters A, B, C… to the criteria keys in the order they are given. It reads the letters'
  log-probabilities (one token, `temperature=0`), divides them by `READOUT_T` and applies a softmax.
- The reply's `model` field is a fingerprint string: directory, temperature, flags, shim hash.
- The shim works with **any** open model, so the stock base can be scored through the exact same readout.

## 2. Design and pre-registered hypotheses

**A 2 × 2 design** separates the effect of the *weights* (decision tuning) from the effect of the *readout* (the shim's
prompt layout and temperature against our pipeline):

| | **shim** (vLLM + `shim.py`, `READOUT_T = 0.85`) | **our pipeline** (`hf_engine`, letter logits, our chat prompt) |
|---|---|---|
| **Qwen3.8-27B-FP8** (stock base) | `qwen3.8-27b-shim` *(new)* | `qwen3.8-27b-fp8` *(done)* |
| **OpenJev-FP8** (decision-tuned) | `openjev-27b` *(new, primary)* | `openjev-27b-hf` *(new)* |

**Common to every cell:**
- the same 1,038 clauses and 9 texts (plain, marked, 7 perturbations);
- the same 3 rotated option orders and geometric mean;
- the same folds, calibration, B3 fusion (w = 0.5) and group bootstrap.

**The primary contrast is the left column**: the weights differ, and nothing else does.

**Pre-registered hypotheses.** All use plain text; paired group bootstrap, 5,000 draws; Holm over H1–H3.

| id | hypothesis | test | supports "tuning fixes how, not what" if… |
|---|---|---|---|
| **H1** | Decision tuning improves the raw answers | macro-F1(openjev-27b raw) − macro-F1(qwen3.8-27b-shim raw) | interval > 0 |
| **H2** | Decision tuning does not add usable knowledge | the same difference, on calibrated rows | interval inside **±0.03** (equivalence) |
| **H3** | Decision tuning does not change complementarity with the cues | (B3 + openjev) − (B3 + qwen-shim), calibrated, w = 0.5 | interval inside ±0.03 |

**Secondary (reported with intervals, no correction):**
- **S1 · Order consistency:** the share of clauses whose three orders agree, OpenJev against the base. OpenJev was
  trained for this, so its share should be higher.
- **S2 · Cue reliance:** position and main-clause aspect shares in the B3 refit, against the base and against gold.
  Does tuning move cue *strength* towards corpus *validity*?
- **S3 · Fronting shift and masking contrasts:** does tuning change the order-of-mention reaction?
- **S4 · Readout effect:** the right column against the left. How much of a score is the readout?
- **S5 · Jev:** OpenJev against Jev, open against closed decision training (pipelines differ, so descriptive only).

**Outcomes and wording:**
- **H1 ✓, H2 ✓, H3 ✓:** "Decision tuning makes raw answers usable without changing the information they carry."
- **H1 ✓, H2 ✗ (calibrated OpenJev better):** "Decision tuning also adds knowledge, or at least better-ranked
  probabilities."
- **H1 ✗:** "On this task, decision tuning gives no measurable benefit over the base model."

## 3. Code changes (local, tested before anything runs)

1. **`score_http`: model identity for non-TypeSafe servers.** The shim's reply `model` is a fingerprint, not the
   requested id, so the current check (`answered != model` → error) would stop the run.
   - For the TypeSafe origin, keep the strict check.
   - For other servers, require every answer's `model` to equal the **first** one seen in the run (it must stay
     constant) instead of equalling `--id`.
   - Add a test with a fake shim that changes its fingerprint mid-run.
2. **`10_decision.py`: record the server.** For `--engine http` against a non-TypeSafe URL, call `GET <base>/v1/version`
   once and store `model_dir`, `T`, `shim_sha256` and `flags` in `meta.json`. A rerun with a different shim
   temperature or a different shim file then gets a meta mismatch and stops, instead of mixing caches.
3. **No other pipeline changes.**
   - The option names (`before`/`after`/`same_time`) become the shim's `[A] before: …` lines.
   - `hf_engine` already takes any `--id`, which covers `openjev-27b-hf`. The model is tagged `qwen3_5`, and the
     Qwen3.8-FP8 checkpoint already loads, so loading should work. That is confirmed in the smoke test (§5).
4. **Run `pytest`** (decision tests plus the new one).

## 4. Deucalion setup

1. **Storage (needs approval).** Free space was about 24 GB in the project's 1,000 GB quota, and OpenJev-FP8 is about
   29 GB. Delete the **Qwen3.6-27B-FP8** weights (about 30 GB). Its T5 cache is complete and copied locally, and it can
   be re-downloaded. Check `df` afterwards.
2. **Download on the login node** (internet access), into `../models/`, not the HF cache, so the shim and vLLM get
   plain paths:
   - `hf download openjev/openjev-FP8 --local-dir ../models/openjev-FP8`
   - `hf download openjev/openjev --include "helper/*" "NOTICE" --local-dir ../models/openjev-helper` (the shim only;
     no BF16 weights)
   - Qwen3.8-27B-FP8 is already in `../hf_cache`. vLLM reads it offline by its repo id, with `HF_HOME` set.
3. **A separate environment, `.venv-openjev`.** OpenJev pins `vllm==0.29.0`, `openai==3.16.2` and `httpx==0.28.1`,
   which must not disturb the main `.venv` (transformers 5.22):
   `python -m venv .venv-openjev && .venv-openjev/bin/pip install "vllm==0.29.0" "openai==3.16.2" "httpx==0.28.1"`
4. **`run_openjev.slurm`** (new, next to `run_experiment.slurm`), parameters `MODEL`, `NAME`, `READOUT_T` (default 0.85):
   - **`#SBATCH`:** account `eehpc-dev-2026d06-183g`, `dev-a100-80` (4 h) or `normal-a100-80`, 1 GPU, 32 CPUs.
   - **Environment:** `HF_HUB_OFFLINE=1`, `HF_HOME=../hf_cache`, `PYTHONNOUSERSITE=1`.
   - **Ports:** derived from the job ID, as in the old `run_llm_baseline.slurm`, so jobs sharing a node never collide.
     vLLM gets `P`, the shim `P+1`.
   - **Start vLLM in the background** (`.venv-openjev`), with the card's flags: `vllm serve $MODEL --served-model-name
     qwen --port $P --max-model-len 16384 --gpu-memory-utilization 0.90 --trust-remote-code --max-num-seqs 16
     --max-logprobs 64 --gdn-prefill-backend triton --quantization fp8`. `--limit-mm-per-prompt` is dropped, since
     the input is text only. Wait for `/v1/models`, up to 20 minutes.
   - **Start the shim** with `VLLM=http://127.0.0.1:$P/v1`, `TOKENIZER=$MODEL` (a local path, so it works offline),
     `SHIM_MODEL=$MODEL`, `READOUT_T`, `--port $((P+1))`. Wait for `/v1/version`.
   - **Run the main pipeline** (`.venv`): `python experiments/10_decision.py --engine http --url
     http://127.0.0.1:$((P+1))/v1/systemone --name $NAME --id qwen`.
   - **Cleanup:** a `trap` kills both servers on exit, success or failure.
5. **The `openjev-27b-hf` cell** uses the existing `run_experiment.slurm`:
   `--name openjev-27b-hf --id <local path to openjev-FP8>` (`10_decision.py` takes a path as `--id`; check the pinned
   revision logic for local paths in the smoke test).

## 5. Runs

| step | job | partition, time | check before moving on |
|---|---|---|---|
| 1 | smoke: `openjev-27b`, `--smoke` (100 rows, writes nothing) | dev-a100-80, 1 h | vLLM loads FP8 on A100 (Marlin); 0 NaN; the fingerprint is constant; throughput |
| 2 | smoke: `qwen3.8-27b-shim` and `openjev-27b-hf`, `--smoke` | dev-a100-80, 1 h each | the same checks; `hf_engine` loads OpenJev |
| 3 | full: `openjev-27b` | dev-a100-80, 4 h (about 1 h expected; a 9B decision run took 27 min) | all 9 texts cached |
| 4 | full: `qwen3.8-27b-shim`, `openjev-27b-hf` | dev-a100-80, 4 h each | as above |
| 5 | locally: copy the caches back, rebuild `t5_decision.csv` (`10_decision.py`, no `--model`), compute H1–H3 and S1–S5 | Mac, about 20 min | — |
| 6 | app: add the rows to the *LLM decisions* model table; write the outcome on *Conclusions*; add a row to *Project summary* | | |

**Resumes are safe:** a cache is written per text, and the meta check stops mixed settings. Expected GPU cost is
about 6 GPU-hours. Expected API cost: none.

## 6. Risks and fallbacks

| risk | fallback |
|---|---|
| vLLM 0.29 FP8 kernels don't run on A100 (no native FP8) | vLLM uses Marlin weight-only FP8 on Ampere. If that fails, run only the `hf_engine` column (OpenJev through our pipeline) and report H1–H3 on that column |
| `--gdn-prefill-backend triton` or other flags are unknown to the installed vLLM | drop the flag; record the actual command in the job log |
| The shim needs the internet for the tokenizer | `TOKENIZER` points to the local model directory |
| Quota is still too tight after removing Qwen3.6 | also remove the 4B or 9B weights (both cached); or ask Deucalion support for more |
| The OpenJev Flash 9B base is unknown | optional extra cell only after reading its `NOTICE`; matched to its base if that is Qwen3.5-9B |
| Licence | CC BY-NC 4.0: research only; cite OpenJev and its base in the paper; no redistribution of weights |

## 7. What changes in the paper

The T5 section gains a **controlled test of decision tuning**: same base weights, same readout, tuned or not. It
also answers the reviewers' demand for an open replication of the Jev result (EIC W2; R3 W3). It's also the clearest
way to say what decision models contribute to linguistic analysis: better-behaved answers, more knowledge, or both.

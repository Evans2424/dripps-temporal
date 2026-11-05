# RQ2 pre-registration: does a model's training variety matter?

**Status:** written 2026-11-05, before any grid model was run. **Frozen at commit:** `________` (fill in before the
full runs; nothing below changes after that commit).

## Question

In Table 6, zero-shot LLMs are weakest on the Angolan (AP) and Mozambican (MP) varieties: Qwen3.8-27B raw scores
0.475 on AP against 0.578 on EP, while CueLR is strongest there. We ask:
- Does a model score higher on the variety it was trained on?
- Do models trained on European Portuguese transfer better to AP and MP, whose written norm is closer to EP's?

## Models

| role | model (Hugging Face id) | training variety |
|---|---|---|
| **confirmatory LLM pair** | `PORTULAN/gervasio-7b-portuguese-ptpt-decoder` / `PORTULAN/gervasio-7b-portuguese-ptbr-decoder` | PT-PT / PT-BR. Same Llama-2-7B base, same recipe and size |
| **confirmatory encoder pair** | `PORTULAN/albertina-100m-portuguese-ptpt-encoder` / `PORTULAN/albertina-100m-portuguese-ptbr-encoder` | PT-PT / PT-BR. Same architecture and recipe |
| exploratory, PT-PT | `amalia-llm/AMALIA-9B-0626-DPO`, `PORTULAN/gervasio-8b-portuguese-ptpt-decoder` | PT-PT |
| exploratory, PT-BR | `recogna-nlp/bode-7b-alpaca-pt-br-no-peft`, `maritaca-ai/sabia-7b` (base LM), `TucanoBR/Tucano-2b4-Instruct` | PT-BR |
| multilingual controls | `utter-project/EuroLLM-9B-Instruct`, `BSC-LT/salamandra-7b-instruct`, `Qwen/Qwen3.5-9B` | — |

The revision of each model is pinned in its cache's `meta.json` by the first run.

## Protocol

The protocol is the T5 zero-shot protocol (`experiments/10_decision.py`): option-letter probabilities, three cyclic
option orders, debiasing by geometric mean, and the existing pooled 12-parameter calibration on the existing folds
and seeds.

The LLMs differ from T5 in two ways:
- **Prompt.** Every LLM gets template 2 (Portuguese) with the answer cue "Resposta:". The Portuguese-only models are
  instruction-tuned in Portuguese, so an English prompt would penalise the prompt language rather than the variety.
  The wording is neutral between PT-PT and PT-BR.
- **Plain prompt.** Checkpoints without a chat template (Gervásio-7B, Sabiá) get the same user text as a plain
  completion (`render_plain`).

The encoders are run as the `marked` variant, with 5 seeds and the same configuration grid as `marked@albertina`
(`experiments/08_encoder.py --checkpoint albertina-ptbr`).

## Tests (`experiments/13_variety_models.py`)

**Metric:** macro-F1 averaged over seeds. LLMs use calibrated probabilities, encoders their out-of-fold
predictions.

- **H1, matching advantage** (per confirmatory pair):

  [F1(PT-PT, EP) − F1(PT-BR, EP)] − [F1(PT-PT, BP) − F1(PT-BR, BP)] > 0.
- **H2, African transfer** (per confirmatory pair): F1(PT-PT, AP∪MP) − F1(PT-BR, AP∪MP) > 0.

**Inference:**
- a paired bootstrap that resamples sentence groups within each variety (1,000 draws), with the same draws for both
  models;
- one-sided p = (1 + #{draws ≤ 0}) / 1,001;
- Holm correction over the four tests (2 pairs × H1, H2).

**Gate:** a pair is tested only if both models' pooled macro-F1 intervals exclude 1/3 (uniform guessing). Otherwise
we report it as uninformative.

**M1, manipulation check (LLMs, descriptive):** the mean token NLL of each raw sentence, by variety (`nll.npy` in each
cache). The prediction: the PT-PT model's NLL is lower, relative to the PT-BR model's, on EP, AP and MP than on BP.

## Power (computed before any grid run)

The bootstrap SD of the contrasts was estimated on existing stand-in pairs over the same clauses:
- LLMs: Qwen3.5-9B against Qwen3.5-4B, calibrated;
- encoders: Albertina against BERTimbau, `marked`.

The minimum detectable effect is at 80% power, one-sided α = .0125 (Holm's smallest threshold for 4 tests):

| contrast | SD | minimum detectable effect |
|---|---|---|
| H1, LLM pair | 0.054 | **0.17** macro-F1 |
| H2, LLM pair | 0.028 | 0.09 |
| H1, encoder pair | 0.030 | 0.09 |
| H2, encoder pair | 0.016 | 0.05 |

**Consequence:** H1 for the LLM pair can detect only a large matching advantage. A non-significant H1 is reported as
**inconclusive** (its interval is given), never as "no effect".

## Exploratory (no confirmatory claims)

- The model × variety table for every model (`t6_variety_by_model.csv`).
- Variety-specific models against the multilingual controls.
- CueLR + model fusion per variety.
- Calibration with variety intercepts as a sensitivity check, so that base rates are not counted as skill.
- The template-2 against template-1 contrast for Qwen3.5-9B (the effect of prompt language).

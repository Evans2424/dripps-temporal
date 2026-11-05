# Calibration in the T5 decision experiment

This note explains the *calibrated* rows of `results/tables/t5_decision.csv`: what calibration
means, why the decision models need it, how `experiments/10_decision.py` does it, and how to read
the result. Code: `debias` and `calibrate` in `src/dripps/decision.py`, and the fold loop in
`experiments/10_decision.py`. The figures below come from the runs of 4 November 2026 (Qwen3.5-4B and
9B on all texts, Jev 1.14 on the plain text). The 27B runs and the rest of the Jev run will add rows,
but the method does not change.

## 1. What a calibrated probability is

A model is **calibrated** when its probabilities can be taken at face value. Of all the clauses where
it says "Ant, 0.8", about 80% should really be Ant. Accuracy and calibration are separate properties:

- A model can be accurate and badly calibrated: right 80% of the time while always reporting 0.99.
- A model can be calibrated and useless: always reporting the corpus base rates (0.25 / 0.42 / 0.33)
  is perfectly calibrated and never better than the majority class.

Calibration matters here for three reasons:

1. **Comparison across families.** B3, the encoders and the LLMs are compared on probability
   metrics as well as macro-F1. A model that ranks the readings well but reports nonsense numbers
   would look worse on log loss than it is.
2. **Fusion.** The `B3 + decision … w=0.5` rows average two probability vectors. That only makes
   sense if both are on the same scale.
3. **The perturbations (T3).** The perturbation table reports changes in the probability of the
   gold reading. A shift of 0.1 means something only if 0.1 is a real change in likelihood.

### How calibration is measured

The table reports three numbers for every row (`prob_metrics` in `10_decision.py`), pooled over
seeds:

| metric | definition | what it rewards |
|---|---|---|
| `log_loss` | mean of −log P(gold reading) | a proper scoring rule [gneiting2007]: punishes confident mistakes hard |
| `brier` | sum over the 3 readings of (P − 1[gold])², averaged | also proper [brier1950], and bounded, so one mistake can't dominate |
| `ece10` | expected calibration error on the top probability, 10 equal-width bins [naeini2015ece] | calibration alone: in each bin, the gap between the mean top probability and the share correct, weighted by bin size |

Log loss and Brier mix accuracy and calibration. ECE isolates calibration but says nothing about
accuracy (the base-rate model above has ECE ≈ 0). Read them together.

## 2. Why the decision models are miscalibrated

Nothing in T5 is trained on this task. The Qwen models and Jev are read **zero-shot**: one forward
pass, three option probabilities (see the "How it works" tab of the demo page). There are several
known reasons why those probabilities are not calibrated for *this* task.

### 2.1 The training objective is about text, not about this question

A language model's next-token distribution is fitted to its pretraining text. The probability it
gives the token " A" after our prompt is the probability of that token in text like this, not the
probability that the participial clause is anterior. Base models are reasonably calibrated on
multiple-choice questions with familiar formats [kadavath2022]. But instruction tuning and RLHF are
known to make them overconfident: GPT-4's calibration on MMLU got markedly worse after
post-training [openai2023gpt4]. The Qwen3.5 models used here are instruction-tuned chat models.

Large neural classifiers in general tend to be overconfident, which is the setting in which
temperature scaling was proposed [guo2017calibration].

### 2.2 Prompt-induced biases: majority-label, recency and surface-form bias

Zhao et al. [zhao2021calibrate] showed that an LLM's answer distribution depends on the prompt
itself, not only on the input:

- **Majority-label bias:** a pull towards whichever answer the prompt format makes most likely.
- **Recency bias:** a pull towards the last option.
- **Common-token bias:** a pull towards answers that are frequent tokens.

Their remedy, *contextual calibration*, is a per-class correction of exactly the kind section 4 fits.

**What we see.** Qwen3.5-4B and 9B pick whichever slot holds **Ant** for about 98% of the clauses,
in all three option orders (`t5_decision_diag.csv`). This is a label bias, not a position bias,
because it follows the reading wherever the reading is placed. A plausible reason, *not tested here*:
the *tendo* + past participle construction is morphologically a compound perfect, and the perfect
is the textbook marker of anteriority. So "Before" is the surface reading a model with shallow
grammatical knowledge would give. The corpus disagrees: Post is the most frequent reading (42%) and
Ant the least frequent (25%).

### 2.3 Option-order (selection) bias

LLMs answering multiple-choice questions prefer some option letters regardless of content
[zheng2024selection; pezeshkpour2024order]. This is not a calibration problem in the narrow sense,
and it is handled separately and without labels (section 3).

### 2.4 Jev: trained for calibrated decisions, but not for this task

TypeSafe trains Jev specifically to return calibrated probabilities (its docs call this
*Reinforcement Learning for Calibrated Decisions*; the method is not published). Calibration is
always relative to a distribution, though: a model calibrated on TypeSafe's training decisions need
not be calibrated on Portuguese participial clauses, whose base rates it has never seen.

**What we see (plain text, raw).**
- **Overconfident:** the mean top probability is 0.70, but accuracy is 0.625. ECE is 0.084, against
  0.494 for raw Qwen 9B and 0.020 for B3.
- **Skewed between readings:** it predicts Simul for 45% of clauses (gold 33%) and Post for 27%
  (gold 42%).

So Jev is much closer to calibrated than the stock models, but not calibrated for this corpus.

### 2.5 Clipping

2.4% of Jev's probabilities are exactly 0, as the API's own examples show it can return. The log of 0
is undefined, so `calibrate` clips probabilities to at least 1e-6 (log = −13.8) before taking logs.
This caps how strongly a "certain" answer can count. No Qwen probability falls below the clip.

## 3. Step one, without labels: cancelling position bias

Before any calibration, each clause is scored **three times**, with the readings rotated through the
slots (`CYCLES` in `decision.py`):

```
order 0:  A = Ant    B = Post   C = Simul
order 1:  A = Post   B = Simul  C = Ant
order 2:  A = Simul  B = Ant    C = Post
```

Every reading occupies every slot exactly once. `debias` takes the **geometric mean** of the three
probability vectors (the mean of the log-probabilities) and renormalises.

**Why this cancels position bias exactly.** Suppose a slot preference adds a constant `c_s` to the
log-probability of whatever sits in slot `s`:

```
log P_k(reading j) = a_j + c_slot(j, k) + const_k
```

Averaging over the three orders gives `a_j + (c_A + c_B + c_C)/3 + const`. The slot term is the same
for every reading, so it disappears when the vector is renormalised. An *arithmetic* mean of the
probabilities would not cancel it, because the slot effect is multiplicative in probability space.
Using all six permutations would give the same guarantee at twice the cost.

This step uses no labels, so the **raw** rows stay fully zero-shot. It removes only biases that
depend on the slot. A bias that follows the reading itself, like Qwen's pull towards Ant, survives,
and that is what the next step addresses.

## 4. Step two, with labels: Dirichlet calibration on each training fold

### 4.1 The model

`calibrate(P_tr, y_tr, P_te)` fits a multinomial logistic regression whose inputs are the
**log-probabilities** of the three readings:

```
z      = log(clip(P, 1e-6, 1))                 # 3 numbers per clause, after debiasing
P_cal  = softmax(W z + b)                      # W: 3×3, b: 3  →  12 parameters
```

This is **Dirichlet calibration** [kull2019dirichlet]: multinomial logistic regression on
log-probabilities. It is the multiclass generalisation of Platt scaling [platt1999] and contains the
simpler methods as special cases:

| method | constraint on (W, b) | what it can fix |
|---|---|---|
| temperature scaling [guo2017calibration] | W = (1/T)·I, b = 0 | over- or underconfidence only |
| contextual / bias calibration [zhao2021calibrate] | W = I, free b | a constant pull towards one reading |
| vector scaling | W diagonal, free b | the above, with one scale per reading |
| **Dirichlet (ours)** | W, b free | also systematic confusions between two readings |

**Why not something simpler.** Temperature scaling has one parameter and cannot change which
reading wins, so it could never correct Qwen's "always Ant". The intercepts `b` are what can.

**Why not something richer.** Isotonic regression [zadrozny2002] or histogram binning would need
far more data per fold than ~830 training clauses, and they do not extend cleanly to three classes.
With 12 parameters, the calibrator is small next to B3, which fits one coefficient per cue and
reading.

### 4.2 Implementation details

- **Library:** scikit-learn's `LogisticRegression(max_iter=1000)` [pedregosa2011]: L2 penalty with
  C = 1, lbfgs solver, multinomial loss. The penalty is light at this sample size; it mainly keeps W
  finite when a model's log-probabilities are nearly collinear. Kull et al. recommend a different,
  off-diagonal penalty (ODIR); we kept the default for simplicity.
- **Which probabilities are calibrated:** the debiased plain-text and marked-text probabilities,
  separately. The perturbation table (T3) uses **raw** probabilities, so its shifts are not affected
  by calibration.
- **Folds:** each calibrator is fitted on the **training rows of one fold** and applied to that
  fold's **test rows**. The folds are B3's: `StratifiedGroupKFold`, 5 splits, grouped by sentence so
  that rows sharing a sentence never straddle train and test, and repeated for 5 seeds. Every
  calibrated probability is therefore out-of-fold, like every B3 prediction. The calibrator never
  sees the gold label of the clause it scores.
- **Seeds:** the raw LLM probabilities are deterministic and identical across seeds. Only the fold
  assignment changes, so `sd_seed` for calibrated rows measures sensitivity to the split.

### 4.3 What calibration can and cannot do

It **can**:
- rescale confidence (fix over- or underconfidence);
- remove a constant pull towards a reading (Qwen's Ant bias, Jev's Simul bias);
- correct a systematic confusion, such as "when the model says Simul it is often Post";
- change the top reading. That is why calibrated macro-F1 differs from raw macro-F1: it is not just
  a cosmetic change to the numbers.

It **cannot**:
- add information the model's three numbers do not contain. It is one global map, the same for
  every clause, and cannot fix a clause-specific error;
- tell you anything about a clause that the raw probabilities ranked wrongly *and* atypically.

## 5. What the results show so far

| row | macro-F1 | log loss | Brier | ECE |
|---|---|---|---|---|
| B3 logit | 0.694 | 0.761 | 0.434 | 0.020 |
| Qwen3.5-4B, raw | 0.157 | 3.215 | 1.373 | 0.693 |
| Qwen3.5-4B, calibrated | 0.474 | 0.973 | 0.586 | 0.016 |
| Qwen3.5-9B, raw | 0.162 | 1.537 | 0.972 | 0.494 |
| Qwen3.5-9B, calibrated | 0.581 | 0.865 | 0.512 | 0.018 |
| Jev 1.14, raw (plain) | 0.628 | 0.850 | 0.507 | 0.084 |
| Jev 1.14, calibrated | *pending* | | | |
| B3 + Qwen3.5-9B calibrated, w=0.5 | 0.721 | 0.737 | 0.424 | 0.122 |

How to read it:

1. **Calibration recovers hidden signal in the stock models.** Raw Qwen 9B is below chance (0.16)
   because its top answer is almost always Ant. Its probabilities still rank the readings sensibly
   relative to each other, and 12 parameters fitted on training folds lift it to 0.58. The pull
   towards Ant was a constant offset on top of real information.
2. **After calibration, ECE drops to B3's level** (about 0.02). For the Qwen models, the remaining gap
   to B3 is about discrimination, not calibration.
3. **Jev raw is already better than Qwen 9B calibrated.** Its remaining miscalibration (ECE 0.084,
   overconfident, Simul-heavy) is the kind section 4 corrects. Its calibrated row will show how much
   of the remaining gap to B3 is bias rather than missing knowledge.
4. **Fusion is not calibrated.** Averaging two calibrated distributions produces a mixture that is
   systematically underconfident, so ECE rises to about 0.12–0.15 for every `B3 + …` row, even
   though macro-F1, log loss and Brier improve. For B3 + Qwen 9B, the mean top probability is 0.61
   while accuracy is 0.73. Report fused rows for macro-F1. If fused
   probabilities are ever used directly, they need their own calibration step.

## 6. How to report it

- **The raw row is the zero-shot claim.** "Qwen 9B answers Before for 98% of clauses" and "Jev
  reaches 0.63 macro-F1 without labels" are statements about the raw rows.
- **The calibrated row is a supervised probe with 12 parameters.** It answers "how much usable
  signal do the model's probabilities contain?", not "what does the model know unaided?". Say that
  it uses the training folds' labels.
- **Don't mix raw and calibrated rows across models in one comparison.** Compare raw with raw and
  calibrated with calibrated.
- **For Jev, the raw-versus-calibrated gap tests TypeSafe's claim.** A small gap means its
  probabilities are calibrated for this task; a large one means they are not.

## References

Keys refer to `docs/references.bib`. The calibration entries were added for this note.

- [brier1950] Brier (1950). Verification of forecasts expressed in terms of probability.
- [gneiting2007] Gneiting & Raftery (2007). Strictly proper scoring rules, prediction, and estimation.
- [guo2017calibration] Guo, Pleiss, Sun & Weinberger (2017). On calibration of modern neural networks.
- [kadavath2022] Kadavath et al. (2022). Language models (mostly) know what they know.
- [kull2019dirichlet] Kull et al. (2019). Beyond temperature scaling: Dirichlet calibration.
- [naeini2015ece] Naeini, Cooper & Hauskrecht (2015). Obtaining well calibrated probabilities using Bayesian binning.
- [openai2023gpt4] OpenAI (2023). GPT-4 technical report.
- [pedregosa2011] Pedregosa et al. (2011). Scikit-learn: machine learning in Python.
- [pezeshkpour2024order] Pezeshkpour & Hruschka (2024). Large language models sensitivity to the order of options in multiple-choice questions.
- [platt1999] Platt (1999). Probabilistic outputs for support vector machines.
- [zadrozny2002] Zadrozny & Elkan (2002). Transforming classifier scores into accurate multiclass probability estimates.
- [zhao2021calibrate] Zhao, Wallace, Feng, Klein & Singh (2021). Calibrate before use.
- [zheng2024selection] Zheng, Zhou, Meng, Zhou & Huang (2024). Large language models are not robust multiple choice selectors.

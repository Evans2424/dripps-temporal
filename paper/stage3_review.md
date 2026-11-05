# Stage 3: full panel review of draft 1

**Panel:**
- EIC: ACL area chair, Semantics.
- R1, methodology: statistics and NLP evaluation.
- R2, domain: Portuguese tense and aspect.
- R3, perspective: psycholinguistics and LLM evaluation.
- DA: Devil's Advocate.

**Date:** 2026-11-05. **Read-only:** no manuscript file is changed by this review.

## EIC: Minor Revision (confidence 4)

A tight, well-evidenced computational paper. The six-move story reads well, and the paired, group-bootstrapped design
is above venue standard.
- **W1:** contribution (iii) and part of (ii) are placeholders. Acceptable at draft stage, but the paper must be
  judged on the delivered parts: make clear which claims are confirmatory now.
- **W2:** there are no figures. A visual for the variety pattern or the cue strength would help skimming reviewers.
- **W3:** the abstract is dense (a 180-word block); one sentence could go.

## R1, methodology: Minor Revision (confidence 5)

**Strengths:**
- paired comparisons on shared folds throughout;
- a fusion weight fixed in advance;
- raw LLM rows that use no labels;
- every number reproducible from scripts.

**Weaknesses:**
- **W1 (Major): multiplicity in Table 4.** Four of five gains have intervals above 0, with lower bounds of 0.008–0.018.
  State which one is confirmatory (the pre-set headline, Qwen3.8-27B) and that the others are exploratory; the
  methods say this, but the results don't.
- **W2 (Minor): "below chance" is undefined.** Give the reference value. Uniform random guessing gives macro-F1 ≈ 0.33;
  the majority class gives 0.196.
- **W3 (Minor): Table 6's B3 row.** It is computed from saved predictions (B3 recovered from the fused probabilities).
  Say so in the caption or the appendix.
- **W4 (Minor): the identical-cue measure is uncalibrated across models**, since it compares raw probabilities. That is
  fine for within-model contrasts; say that the Qwen and Jev values aren't directly comparable.

## R2, domain: Minor Revision (confidence 4)

**Strengths:** the (1a–c) triplet is an effective hook; the null result for participial aspect is backed by checks on
the cue's own variation.

**Weaknesses:**
- **W1 (Minor): genre and topic confound across varieties.** AP and MP texts may differ in topic (sports and politics
  reporting with dated sequences), which could explain both the Post-heavy base rates and the LLM drop. Add this to
  Limitations.
- **W2 (Minor): the Discussion's suggestion of "lexical" new cues is speculative.** Phrase it as a hypothesis for the
  annotation scheme.

## R3, perspective: Minor Revision (confidence 3)

The cue-strength against cue-validity contrast is the paper's most transferable idea.
- **W1 (Major): no human baseline.** Without untrained native-speaker judgements, a raw 0.56–0.63 can't be placed.
  Add it as planned work in Limitations.
- **W2 (Minor):** the paper could note that order of mention is a known default in discourse interpretation. A
  one-clause mention suffices, given the stated emphasis on computation.

## Devil's Advocate

**Strongest counter-argument.** The "+0.043" gain may say little about *temporal* knowledge. Any second classifier
with different errors improves an ensemble, and LLMs are simply more different from B3 than encoders are, since
encoders are trained on the same labels.

**Answer available in the paper:** the encoders *also* make different errors (marked encoders score about B3's level
from text only), yet fusion with them gains nothing. That argues for different *information*, not merely different
noise.

**Remaining risk:** the LLMs' complementary signal could track annotator idiosyncrasy (there is no IAA). The paper
names this.

| # | severity | issue |
|---|---|---|
| DA1 | MAJOR | Annotation noise as the alternative explanation of the LLM gain, without IAA (stated in Limitations; robustness test B4 planned) |
| DA2 | MAJOR | Contribution (iii) is announced in the introduction but pending; the reader must not mistake it for a result |
| DA3 | MINOR | The 98% "Ant" collapse of small models is reported without saying whether calibration fully fixes it (it does: calibrated 4B/9B reach 0.48/0.58) |

No CRITICAL issue.

## Editorial decision: **Minor Revision**

**Consensus:**
- strong design;
- the central result (an LLM gain over a cue model that encoders lack) is well supported for the pre-set headline;
- the pending parts must be clearly separated from the delivered ones.

**Revision roadmap:**

| id | source | action |
|---|---|---|
| RV1 | R1-W1, EIC-W1 | Name the confirmatory comparison in §5.3; mark the others exploratory |
| RV2 | DA2, EIC-W1 | In the introduction's contributions, mark what is delivered and what is designed/pending |
| RV3 | R1-W2 | Define chance and majority references for macro-F1 |
| RV4 | R1-W3 | Table 6 caption: how B3's per-clause predictions are obtained |
| RV5 | R1-W4 | Note that the identical-cue values are within-model, not comparable across models |
| RV6 | R2-W1, R3-W1 | Limitations: genre/topic confound across varieties; no native-speaker baseline (planned) |
| RV7 | R2-W2 | Discussion: phrase the lexical cues as a hypothesis |
| RV8 | DA3 | One clause on what calibration does for the small models |
| RV9 | EIC-W2 | Figure: **declined for this round.** The ~2.3 free pages are reserved for the pending results (OpenJev, variety-matched models), whose tables already have their slots; add a figure once those land |
| RV10 | EIC-W3 | Trim the abstract by one sentence |

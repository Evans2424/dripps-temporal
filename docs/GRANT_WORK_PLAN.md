## Overview

The project asks how well three families of models (fine-tuned encoders, generative language models and decision-making models) recover the temporal relation of Portuguese adverbial perfect participial clauses, and whether what they rely on agrees with linguistic theories of aspect and temporal interpretation. The relation is anteriority, posteriority or simultaneity between the clause (*tendo* plus a participle) and its main clause. Classical ML methods are already implemented and run on the annotated clauses of the DRIPPS corpus, nominately a multinomial logit, random forest and XGBoost, and a hierarchical varieties model for RQ2. They serve as baselines for the other models that we will experiment with.

## Phase 1 (months 1 and 2): fine-tuned encoders

Question: does the sentence text alone recover the signal carried by the human-coded cues, and which parts of the sentence does an encoder use?

1.1 Fine-tune XLM-R (base and large), mDeBERTa and a Portuguese-specific encoder with the sentence as the only input; no annotation column is visible to the model. Hyperparameters are chosen on a grouped split inside each training fold, never on the held-out fold. One tuned configuration takes about 250 fine-tuning runs, roughly 4 to 6 hours on one L40-class GPU.

1.2 Train a marked variant in which the participial group and the main verb are delimited in the text and pooled at the verbs. Compare it with the plain variant.

1.3 Concatenate the annotated features (main-clause aspect and tense, clause position, connector, participial-clause aspect) with the encoder embedding before the classification layer, for both the plain and the marked variants. Compare with the text-only encoder and with the logit. This tests whether text adds information beyond the annotated cues, and whether the encoder uses the cues once they are given. As a simpler check, also average encoder and logit probabilities (fusion).

1.4 Run the masking and displacement experiments. Mask the participle, the main verb or the temporal adverbials, or move the participial clause to the front of the sentence. 

Output: tuned predictions, ablation tables and paired comparisons with the classical baselines.

## Phase 2 (months 2 to 4): generative models

Question: how do open-weight language models classify the relation, and do they react to perturbations as the encoders and the theory predict?

2.1 Prompt four to five instruction-tuned open-weight models (7B to 70B) zero-shot and few-shot on the same folds. The output is constrained to the three labels.

2.2 Apply the Phase 1 perturbations, so that shifts in predicted probability can be compared across families.

2.3 Prompting engineering, for example, adding the annotated features to the prompt.

2.4 Fine-tune two to three of these models with LoRA or QLoRA on the training folds (month 4).

This phase takes most of the GPU budget.

## Phase 3 (months 3 to 5): decision-making models

Question: how does a Jev-style decision model behave on the same task and perturbations?

3.1 Choose two open reproductions of the mechanism, for example parallel constrained decoding on a stock model and a text-diffusion approach, and compare them.

3.2 Adapt the label set and scoring to the three-way task and evaluate on the same folds with the same metrics.

3.3 Run the Phase 1 perturbations.

## Phase 4 (months 4 to 6): comparison with linguistic theory

Question: do the models rely on the cues that the theories name, and do the three families agree with each other?

4.1 State the main theoretical claims as predictions about the cues. Telic situations go with anteriority or posteriority, durative situations with simultaneity, and the main clause carries more of the reading than the participial clause.

4.2 For each family, test these predictions with the ablation and displacement contrasts, and measure agreement between families on the same sentences.

4.3 Address the variety question (RQ2): whether the association between cues and readings differs across the four varieties once differences in base rates are accounted for, and whether the models show the same pattern.

4.4 Read the disagreements by hand to find which sentence types separate the models from the classical analysis.

Output: comparison tables and a list of claims that are confirmed, qualified or contradicted.

## Phase 5 (months 5 and 6): dissemination

5.1 Write a paper for an international computational linguistics venue.

5.2 Publish code, configuration and result tables in a public repository with pinned dependencies.


## Human resources

A principal investigator in computational linguistics, a researcher in Portuguese linguistics who interprets the aspectual and temporal categories, and a machine learning researcher or doctoral student who runs the experiments. Names and time commitments come from the proposal.


## Task relations and schedule

| Months | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| Phase 1, encoders | x | x | | | | |
| Phase 2, generative models | | x | x | x | | |
| Phase 3, decision models | | | x | x | x | |
| Phase 4, theory comparison | | | | x | x | x |
| Phase 5, dissemination | | | | | x | x |



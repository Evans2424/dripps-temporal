### 1. Experiments already performed

The classical models (multinomial logit, random forest, XGBoost and a hierarchical varieties model) run on the 1,038 Portuguese clauses with 5-fold cross-validation over five seeds, in minutes on a laptop CPU. The encoder pipeline is implemented and tested on CPU, and a first fine-tuning run of XLM-R base on a lab GPU showed that tuning is needed (macro-F1 0.469 against 0.694 for the logit). Generative and decision-making models have not been run yet.

### 2. Phases and time management

The project lasts 6 months, and all phases use the same folds and perturbations.

- Weeks 1 and 2: set up the environment on Deucalion, download the models and time one fold of each family.
- Months 1 and 2: fine-tuned encoders, with feature concatenation and masking runs.
- Months 2 to 4: generative models, zero-shot, few-shot and LoRA.
- Months 3 to 5: decision-making models.
- Months 4 to 6: analysis, on CPU and outside the cluster.

Each job is one (model, variant, seed, fold) unit in an independent job array, so a failure is cheap and a queue delay in one phase does not block the others.

### 3. Computational resources

| Item | Estimated GPU hours |
|---|---|
| Encoders (XLM-R base and large, mDeBERTa, a Portuguese encoder; 4 configurations each) | 140 to 190 |
| Generative models (4 to 5 open-weight models, 7B to 70B; prompts and perturbations) | 250 to 400 |
| LoRA fine-tuning of 2 to 3 models | 80 to 120 |
| Decision-making models (2 open reproductions) | 100 to 150 |
| Pilot and development | 30 |
| Total, with a 30% margin | about 1,000 |

The encoders fit on one GPU, and no job needs more than one node. The 70B model needs several GPUs or quantisation, and the pilot settles this. CPU use is small.

### 4. Users and training

Three users: the principal investigator, a researcher in Portuguese linguistics, and the machine learning researcher or doctoral student who submits the jobs and is the only one needing cluster access. The team already works with the Python pipeline. The only training needed is the Deucalion onboarding (scheduler, partitions, job arrays) and serving a large model on several GPUs, both covered in the pilot.

### 5. Storage and data transfer

The data and the outputs are small. Code reaches the cluster with `git pull`, and the corpus is a few megabytes. Model weights (about 1 GB per encoder, 16 GB per 7B model, 140 GB for a 70B model; under 500 GB in total) are downloaded once to shared storage, since compute nodes may lack internet access. Only predictions and tables are saved (a few tens of megabytes), each run writes its own file and skips finished work on restart, and results are copied back and committed with the code. Dependencies are pinned and each table records the commit, seed and model. Scratch files are deleted at the end.

### 6. Risks

- If the 70B model does not fit, use a smaller model with the same prompts.
- If no open reproduction of the decision-making mechanism is stable, keep the best open approximation and state its limits.
- If the queue is slow, partial results from the independent arrays are still usable.

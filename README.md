# Cue hierarchies in Portuguese perfect participial clauses

Code for a COLING 2027 submission analysing how the temporal reading
(anteriority / posteriority / simultaneity) of adverbial perfect participial
clauses is cued across four varieties of Portuguese, using the
[DRIPPS](https://github.com/johnycordeiro/DRIPPS) corpus (993 annotated
sentences; Silvano et al., LDK 2023).

## Questions

- **RQ1** Which cues drive the temporal reading, and what is their relative contribution?
- **RQ2** Do the four Portuguese varieties weight the same cue inventory differently?
- **RQ3** Does a model learned on one variety transfer to the others?

## Setup

```bash
python3.13 -m venv .venv && .venv/bin/pip install -r requirements.txt
make all
```

## Layout

| path | |
|---|---|
| `src/dripps/schema.py` | label inventories; aspect → Moens & Steedman primitives; tense → TAM bundles |
| `src/dripps/io.py` | loader for the unquoted `;`-delimited export, with strict validation |
| `src/dripps/features.py` | the 17-feature model matrix |
| `src/dripps/leakage.py` | blocks the definitionally circular `DR` / `SR-SC` columns |
| `src/dripps/evaluate.py` | grouped CV, repeated folds, bootstrap intervals |
| `src/dripps/results.py` | one place that locates and reads `results/tables/` for the viewer and the app |
| `src/dripps/interpret.py` | block ablation/permutation importance, coefficients, tree rules and structure |
| `src/dripps/explain.py` | TreeSHAP for the two ensemble rungs |
| `experiments/` | `01_audit` `02_baselines` `03_cue_hierarchy` `04_variety_hierarchies` `05_explain` `06_viewer` |
| `docs/` | [plan](docs/PROJECT_PLAN.md) · [scheme](docs/ANNOTATION_SCHEME.md) · [audit](docs/DATA_AUDIT.md) · [findings](docs/FINDINGS.md) · [linguistics](docs/LINGUISTICS.md) · [methods](docs/methods.md) · [deploy](docs/DEPLOY.md) · [refs](docs/references.bib) |
| `results/viewer.html` | generated dashboard — `make viewer`, then open it in a browser |
| `app/` | interactive explorer (Streamlit): corpus filter with mispredicted-only view, model pages, error analysis. `make app`, then open http://localhost:8501; to share it, see [deploy](docs/DEPLOY.md) |

## Three things that will bite you

1. **`DR` and `SR-SC` are definitionally circular with the target.** `SR-SC ==
   "before"` is anterior 100% of the time. Every model path calls
   `assert_no_leakage`; the circular baseline is reported once, deliberately.
2. **Multi-APC sentences are duplicated across rows** — 26 sentences, 55 rows.
   Split on `sentence_group` or the folds leak.
3. **The export is not valid CSV.** 17 sentences contain a literal `;` and
   nothing is quoted; 557 contain non-breaking spaces. `dripps.io` handles both.
   Do not `pd.read_csv` it directly.

## Data

`data/raw/dripps_full.csv` is exported from the DRIPPS desktop application.
The corpus is by Silvano, Cordeiro, Leal & Pais and should be cited as
[`2023.ldk-1.51`](https://aclanthology.org/2023.ldk-1.51/).

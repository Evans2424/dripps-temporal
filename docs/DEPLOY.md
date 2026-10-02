# Deploying the explorer on Streamlit Community Cloud

The explorer in `app/` is a Streamlit app. It runs unchanged on
[Streamlit Community Cloud](https://share.streamlit.io), which builds it from a
GitHub repository.

## What the deployment contains

Community Cloud copies the repository, installs the dependencies, and runs
`streamlit run app/streamlit_app.py` from the repository root. Nothing else runs,
so everything the app reads must be committed:

| the app reads | where it comes from |
|---|---|
| the corpus | `data/raw/dripps_full.csv`, `dripps_violeta.csv`, `dripps_abergaria.csv` |
| every results table | `results/tables/*.csv`, produced by `make all` and committed |
| per-sentence SHAP values | `results/tables/shap_forest.csv`, `shap_xgboost.csv` (about 1.8 MB each, tracked on purpose) |
| the fitted models | refitted on first use and cached; nothing is stored |
| the references | `docs/references.bib` |

Dependencies come from **`app/requirements.txt`**, which Community Cloud reads in
preference to the root `requirements.txt`. It lists only what the app imports.
The root file also carries `shap` (which pulls `numba`) and `statsmodels`, used
by the pipeline and never by the app. Keep the pins in the two files identical,
so the app shows the numbers the pipeline produced.

## Deploy

1. **Push the branch** to the GitHub repository (the one in `git remote -v`).
2. Sign in at <https://share.streamlit.io> with GitHub, grant access to the
   repository, and choose **Create app**.
3. Fill in the form:
   - Repository and branch: the ones you pushed.
   - Main file path: `app/streamlit_app.py`
   - **Advanced settings → Python version: 3.13.** The default is 3.12, the pins
     were tested on 3.13, and the version cannot be set from the repository.
4. **Deploy.** The first build takes a few minutes while the packages install.

## Who can see it

Community Cloud apps are public by default. The corpus is public (DRIPPS,
Silvano et al., LDK 2023) but the results and the error analysis are
unpublished work. To restrict the app, open **App settings → Sharing** and choose
*Only specific people can view this app*, then list the team's email addresses.
Private apps are limited by Community Cloud's plan; check the current limits
before relying on one.

## Behaviour to expect

- **The first visit after a sleep is slow.** Idle apps are put to sleep and
  rebuilt on the next visit. The random forest and XGBoost out-of-fold
  predictions are recomputed then (about ten seconds locally; slower on shared
  CPUs). They are cached afterwards for every visitor.
- **Memory** stays around 250 MB, well under the 1 GB limit.
- **No writes.** The error-review worksheet is held in the browser session; it
  has to be downloaded as CSV before the tab is closed.

## Updating

Push to the deployed branch and the app redeploys. After re-running the
pipeline, commit the changed `results/tables/` files, or the app will show old
numbers beside new code. If `make explain` ran, that includes the two
`shap_*.csv` files.

## Checking before you push

The app was verified from a clean copy of the tracked files, in a fresh virtual
environment built only from `app/requirements.txt`, with every page rendered
from the repository root. To repeat it:

```bash
git ls-files -co --exclude-standard | rsync -a --files-from=- . /tmp/cloud
python3.13 -m venv /tmp/cloud-venv
/tmp/cloud-venv/bin/pip install -r /tmp/cloud/app/requirements.txt
cd /tmp/cloud && /tmp/cloud-venv/bin/streamlit run app/streamlit_app.py
```

If a page fails there, it will fail on Community Cloud too.

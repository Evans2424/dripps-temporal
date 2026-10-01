"""TreeSHAP attributions for the two ensemble rungs.

Writes results/tables/shap_{summary,blocks,forest,xgboost}.csv.

The models are fit on the whole analysis set, the same way 03 fits the logit and
the tree for interpretation. These are descriptions of a fitted model, not
out-of-fold estimates, and do not belong beside the macro-F1 figures in
baselines.csv.

Forest and XGBoost SHAP values are *not* on a common scale: a RandomForest's
leaves hold class probabilities, so its values and base rate live in probability
space, while XGBoost's leaves hold log-odds increments. Compare the two by rank,
never by magnitude.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dripps import explain, features, io, models, schema  # noqa: E402

MODELS = {
    "B5 forest": models.b5_forest,
    "B5 xgboost": models.b5_xgboost,
}


def main() -> None:
    df = io.analysis_frame(io.load())
    X, y = features.build(df), features.target(df)
    ids = df[schema.ID]

    print(f"analysis set: {len(df)} rows, {X.shape[1]} features")

    summaries, blocks = [], []
    for label, factory in MODELS.items():
        print(f"\n{'=' * 78}\n{label} -- TreeSHAP ({explain.PERTURBATION})\n{'=' * 78}")
        fitted = factory().fit(X, y)
        result = explain.tree_shap(fitted, X)

        summary = explain.mean_abs_shap(result, model_label=label)
        summaries.append(summary)

        block = explain.block_shap(result, features.CUE_BLOCKS)
        block.insert(0, "model", label)
        blocks.append(block)

        ranked = (
            block.groupby("block")["mean_abs_shap"].mean()
            .sort_values(ascending=False).round(4)
        )
        print("mean |SHAP| per cue block, averaged over classes:")
        print(ranked.to_string())

        slug = label.split()[-1]
        explain.long_form(result, ids, model_label=label).round(6).to_csv(
            ROOT / f"results/tables/shap_{slug}.csv", index=False)

    pd.concat(summaries).round(6).to_csv(
        ROOT / "results/tables/shap_summary.csv", index=False)
    pd.concat(blocks).round(6).to_csv(
        ROOT / "results/tables/shap_blocks.csv", index=False)

    print("\nwrote 4 tables to results/tables/")


if __name__ == "__main__":
    main()

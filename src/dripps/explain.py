"""SHAP attributions for the tree rungs of the ladder.

Two things about how this is run matter more than the numbers it produces.

**Path-dependent, not interventional.** The interventional estimator marginalises
features independently, which here would score rows with ``both_telic = 1`` while
``mc_telic = 0`` -- combinations ``features.DERIVED`` makes impossible, since
those columns are products of the aspect columns. That is the same failure
``recompute_derived`` exists to prevent in ``block_permutation_importance``. The
path-dependent estimator follows the splits the fitted trees actually took, so it
never leaves the data manifold and needs no background sample.

**These are not out-of-fold numbers.** They describe a model fit on the whole
analysis set, so they say how the fitted model uses a cue, not how well the cue
generalises. Nothing here is comparable to the macro-F1 figures in
``results/tables/baselines.csv``.

The cues are deliberately collinear, and Shapley values split credit between
correlated features rather than assigning it to one. A block whose cues are
redundant therefore looks weaker per feature here than under the block-level
ablation in ``interpret``. Read this as a cross-check on the *ranking*, not as a
competing estimate of importance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PERTURBATION = "tree_path_dependent"


def _underlying(model):
    """The estimator TreeExplainer understands, and the class order to read it by.

    ``models._XGBWrapper`` fits an XGBClassifier on integer codes and exposes the
    real estimator as ``booster_``; a RandomForest is already what we want. Note
    the name: sklearn ensembles define ``estimator_`` as their *unfitted*
    template tree, so unwrapping by that name hands TreeExplainer an unfitted
    DecisionTreeClassifier instead of the forest.

    Either way ``classes_`` on the outer object gives the TR labels in the order
    the per-class output is stacked.
    """
    return getattr(model, "booster_", model), [str(c) for c in model.classes_]


def tree_shap(model, X: pd.DataFrame) -> dict:
    """SHAP values for a fitted tree model, as ``(n_rows, n_features, n_classes)``.

    Returns ``{values, base, classes, features}``. ``base`` is the explainer's
    expected value per class, so ``values[i, :, k].sum() + base[k]`` reproduces
    the model's margin for row *i* and class *k* -- the additivity property the
    regression test pins down.
    """
    import shap

    estimator, classes = _underlying(model)
    explainer = shap.TreeExplainer(estimator, feature_perturbation=PERTURBATION)
    raw = explainer.shap_values(X)

    # shap has returned a list-per-class and a stacked array at different
    # versions, and collapses the last axis when there are two classes
    if isinstance(raw, list):
        values = np.stack(raw, axis=-1)
    else:
        values = np.asarray(raw)
        if values.ndim == 2:
            values = values[:, :, None]

    base = np.atleast_1d(np.asarray(explainer.expected_value, dtype=float))
    if base.shape[0] != values.shape[2]:  # binary case reports a single base
        base = np.repeat(base, values.shape[2])

    return {
        "values": values,
        "base": base,
        "classes": classes,
        "features": list(X.columns),
    }


def mean_abs_shap(result: dict, *, model_label: str = "") -> pd.DataFrame:
    """Mean |SHAP| per feature per class -- the global ranking view."""
    values, classes, features = result["values"], result["classes"], result["features"]
    rows = [
        {"model": model_label, "feature": f, "class": c,
         "mean_abs_shap": float(np.abs(values[:, j, k]).mean())}
        for j, f in enumerate(features)
        for k, c in enumerate(classes)
    ]
    return pd.DataFrame(rows)


def long_form(result: dict, ids, *, model_label: str = "") -> pd.DataFrame:
    """One row per (sentence, feature, class), for the per-sentence views."""
    values, classes, features = result["values"], result["classes"], result["features"]
    ids = list(ids)
    n, n_feat, n_cls = values.shape
    return pd.DataFrame({
        "model": model_label,
        "id": np.repeat(ids, n_feat * n_cls),
        "feature": np.tile(np.repeat(features, n_cls), n),
        "class": np.tile(classes, n * n_feat),
        "shap": values.reshape(-1),
    })


def block_shap(result: dict, blocks: dict[str, list[str]]) -> pd.DataFrame:
    """Mean |SHAP| summed within each cue block, to compare against ablation.

    Summing inside a block before taking the absolute value would let opposing
    per-feature contributions cancel; the block total here is the mean over rows
    of the summed *signed* contribution's magnitude, which is what the block-level
    ablation is closest to.
    """
    values, classes, features = result["values"], result["classes"], result["features"]
    index = {f: j for j, f in enumerate(features)}
    rows = []
    for name, cols in blocks.items():
        js = [index[c] for c in cols if c in index]
        for k, c in enumerate(classes):
            block_total = values[:, js, k].sum(axis=1)
            rows.append({"block": name, "class": c,
                         "mean_abs_shap": float(np.abs(block_total).mean())})
    return pd.DataFrame(rows)

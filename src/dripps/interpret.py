"""Cue-hierarchy extraction.

Importance is measured two ways, because neither alone is sound here.

**Refit ablation** (primary) drops a cue block and refits, answering "does the
model still work without this cue?". It is the only valid measure for the
interaction block, whose columns are deterministic products of the aspect
columns -- permuting those in place either fabricates impossible rows or, once
recomputed, is a no-op by construction.

**Block permutation** (secondary) keeps the fitted model and shuffles a block in
the held-out fold, answering "how much does the fitted model rely on this cue?".
Derived columns are recomputed after each shuffle. Whole blocks are permuted
together so correlated columns cannot mask one another.

Impurity importance is never used: gini rewards high cardinality, and the tense
block expands to seven columns against position's one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold

from .evaluate import macro_f1
from .features import DERIVED_ONLY_BLOCKS, recompute_derived
from .schema import SEED


def _folds(X, y, groups, n_splits, seed):
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(cv.split(X, y, groups))


def _summarise(df: pd.DataFrame, value: str) -> pd.DataFrame:
    out = (
        df.groupby("block")[value]
        .agg(importance="mean", sd="std")
        .sort_values("importance", ascending=False)
    )
    per_fold = df.groupby(["block", "fold"])[value].mean().unstack()
    out["ci_lo"] = per_fold.quantile(0.025, axis=1)
    out["ci_hi"] = per_fold.quantile(0.975, axis=1)
    return out.reset_index()


def block_ablation_importance(
    model, X: pd.DataFrame, y, groups, blocks: dict[str, list[str]],
    *, n_splits: int = 5, seed: int = SEED,
) -> pd.DataFrame:
    """Drop in out-of-fold macro-F1 when a cue block is removed and the model refit.

    Correlated cues can compensate for a dropped block, so this is a test of
    *necessity*: a block scoring near zero is one the model can do without.
    """
    y = np.asarray(y)
    labels = sorted(pd.unique(y))
    records = []

    for fold, (train, test) in enumerate(_folds(X, y, np.asarray(groups), n_splits, seed)):
        full = clone(model).fit(X.iloc[train], y[train])
        base = macro_f1(y[test], full.predict(X.iloc[test]), labels)
        for name, cols in blocks.items():
            keep = [c for c in X.columns if c not in cols]
            # dropping an aspect block leaves its products dangling
            keep = [c for c in keep if c not in _orphaned_derived(cols, keep)]
            est = clone(model).fit(X.iloc[train][keep], y[train])
            s = macro_f1(y[test], est.predict(X.iloc[test][keep]), labels)
            records.append({"fold": fold, "block": name, "drop": base - s, "score": s})
    return _summarise(pd.DataFrame(records), "drop")


def _orphaned_derived(removed: list[str], keep: list[str]) -> set[str]:
    from .features import DERIVED

    return {
        d for d, sources in DERIVED.items()
        if d in keep and any(src in removed for src in sources)
    }


def block_permutation_importance(
    model, X: pd.DataFrame, y, groups, blocks: dict[str, list[str]],
    *, n_repeats: int = 20, n_splits: int = 5, seed: int = SEED,
) -> pd.DataFrame:
    """Drop in out-of-fold macro-F1 when a cue block is permuted in the test fold."""
    y = np.asarray(y)
    labels = sorted(pd.unique(y))
    rng = np.random.default_rng(seed)
    records = []

    for fold, (train, test) in enumerate(_folds(X, y, np.asarray(groups), n_splits, seed)):
        est = clone(model).fit(X.iloc[train], y[train])
        X_test = X.iloc[test]
        base = macro_f1(y[test], est.predict(X_test), labels)
        for name, cols in blocks.items():
            if name in DERIVED_ONLY_BLOCKS:
                continue  # recomputation makes permutation a no-op; use ablation
            Xp = X_test.copy()
            for _ in range(n_repeats):
                order = rng.permutation(len(Xp))
                Xp[cols] = X_test[cols].to_numpy()[order]
                recompute_derived(Xp)
                s = macro_f1(y[test], est.predict(Xp), labels)
                records.append({"fold": fold, "block": name, "drop": base - s, "score": s})
    return _summarise(pd.DataFrame(records), "drop")


def logit_coefficients(fitted_pipeline, feature_names, classes) -> pd.DataFrame:
    """Standardised coefficients, one column per outcome class.

    sklearn returns a single coefficient row for a two-class fit; it is expanded
    to the symmetric two-column form so per-variety and transfer analyses, which
    can hit two-class subsets, behave like the three-class case.
    """
    clf = fitted_pipeline[-1] if hasattr(fitted_pipeline, "__getitem__") else fitted_pipeline
    coef = np.asarray(clf.coef_)
    classes = list(classes)
    if coef.shape[0] == 1 and len(classes) == 2:
        coef = np.vstack([-coef[0], coef[0]])
    return pd.DataFrame(coef.T, index=list(feature_names), columns=classes)


def tree_rules(tree, feature_names, class_names, *, max_depth=None) -> str:
    from sklearn.tree import export_text

    return export_text(
        tree, feature_names=list(feature_names), max_depth=max_depth or 10,
        class_names=list(class_names), decimals=2,
    )

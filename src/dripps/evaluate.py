"""Evaluation protocol.

Three things are enforced here rather than left to each experiment:

* **Splits are grouped by sentence.** Multi-APC sentences are duplicated across
  26 groups / 55 rows, so an ungrouped split leaks.
* **Macro-F1 is computed over a fixed label set.** Rare classes go missing from
  individual folds (British English has 2 ``Post`` rows in total), and without an
  explicit ``labels=`` the average silently ranges over a different number of
  classes per fold.
* **Intervals come from resampling sentence groups, not CV repeats.** Repeats
  differ only by fold seed on identical rows, so bootstrapping them estimates
  the variance of the randomisation, not of the data, and yields intervals an
  order of magnitude too narrow. ``sd_seed`` reports that jitter separately.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold

from .schema import SEED

N_SPLITS = 5
N_REPEATS = 10


def _labels(y) -> list:
    return sorted(pd.unique(np.asarray(y)))


def macro_f1(y_true, y_pred, labels) -> float:
    return float(
        f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    )


def cv_predict(model, X, y, groups, *, n_splits=N_SPLITS, n_repeats=N_REPEATS, seed=SEED):
    """Out-of-fold predictions for each repeat. Returns (n_repeats, n_samples)."""
    Xv = X.to_numpy() if hasattr(X, "to_numpy") else np.asarray(X)
    y = np.asarray(y)
    groups = np.asarray(groups)
    preds = np.empty((n_repeats, len(y)), dtype=object)
    for r in range(n_repeats):
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + r)
        for train, test in cv.split(Xv, y, groups):
            est = clone(model).fit(Xv[train], y[train])
            preds[r, test] = est.predict(Xv[test])
    return preds


def macro_f1_per_repeat(y, preds, labels=None) -> np.ndarray:
    y = np.asarray(y)
    labels = labels or _labels(y)
    return np.array([macro_f1(y, p, labels) for p in preds])


def bootstrap_group_ci(y, preds, groups, *, n_boot=2000, alpha=0.05, seed=SEED):
    """Percentile CI from resampling *sentence groups* with replacement.

    Sentence groups are the independent units, so this reflects how much the
    score would move on a different sample of sentences -- the quantity a
    reader cares about -- rather than how much it moves across fold seeds.
    """
    y = np.asarray(y)
    groups = np.asarray(groups)
    labels = _labels(y)
    rng = np.random.default_rng(seed)

    uniq = np.unique(groups)
    rows_by_group = {g: np.flatnonzero(groups == g) for g in uniq}

    draws = np.empty(n_boot)
    for b in range(n_boot):
        picked = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([rows_by_group[g] for g in picked])
        rep = rng.integers(len(preds))  # fold the seed jitter in too
        draws[b] = macro_f1(y[idx], preds[rep][idx], labels)
    return float(np.quantile(draws, alpha / 2)), float(np.quantile(draws, 1 - alpha / 2))


def score(model, X, y, groups, *, label="", **kw) -> dict:
    preds = cv_predict(model, X, y, groups, **kw)
    labels = _labels(y)
    per_repeat = macro_f1_per_repeat(y, preds, labels)
    lo, hi = bootstrap_group_ci(y, preds, groups)
    return {
        "model": label,
        "macro_f1": float(per_repeat.mean()),
        "ci_lo": lo,
        "ci_hi": hi,
        "sd_seed": float(per_repeat.std(ddof=1)),
        "_preds": preds,
        "_per_repeat": per_repeat,
    }


def per_class_f1(y, preds, labels=None) -> pd.DataFrame:
    y = np.asarray(y)
    labels = labels or _labels(y)
    arr = np.vstack([
        f1_score(y, p, average=None, labels=labels, zero_division=0) for p in preds
    ])
    return pd.DataFrame({"label": labels, "f1": arr.mean(0), "sd": arr.std(0, ddof=1)})


def confusion(y, preds, labels=None) -> pd.DataFrame:
    y = np.asarray(y)
    labels = labels or _labels(y)
    total = sum(confusion_matrix(y, p, labels=labels) for p in preds) / len(preds)
    return pd.DataFrame(total, index=labels, columns=labels)

"""Association statistics for the descriptive audit."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def crosstab(df: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    return pd.crosstab(df[a], df[b])


def cramers_v(table: pd.DataFrame, *, bias_correct: bool = True) -> dict:
    """Cramer's V with Bergsma (2013) bias correction.

    The correction matters here: ``TMC`` has up to 16 levels against
    ``Position``'s 3, and uncorrected V rewards high cardinality, which would
    put tense at the top of the cue hierarchy for the wrong reason.
    """
    chi2, p, dof, _ = stats.chi2_contingency(table)
    n = table.to_numpy().sum()
    r, k = table.shape
    if bias_correct:
        phi2 = max(0.0, chi2 / n - (k - 1) * (r - 1) / (n - 1))
        r_ = r - (r - 1) ** 2 / (n - 1)
        k_ = k - (k - 1) ** 2 / (n - 1)
        denom = min(k_ - 1, r_ - 1)
    else:
        phi2 = chi2 / n
        denom = min(k, r) - 1
    v = float(np.sqrt(phi2 / denom)) if denom > 0 else float("nan")
    return {"v": v, "chi2": float(chi2), "p": float(p), "dof": int(dof), "n": int(n)}


def association(df: pd.DataFrame, cue: str, target: str = "TR") -> dict:
    out = cramers_v(crosstab(df, cue, target))
    out["cue"] = cue
    return out

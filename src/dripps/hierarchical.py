"""B6 helpers: pooled multinomial logit with varying intercepts and slopes by variety.

Main-effect columns are scaled up by ``MAIN_SCALE`` so the L2 penalty barely touches
them; the variety x cue deviation columns are not scaled, so the penalty shrinks them
toward the pooled slope (partial pooling). Varieties are one-hot coded, so every
variety's deviation is shrunk equally (a sum-coded reference level would not be).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

MAIN_SCALE = 10.0
FREE_C = 10.0  # mild ridge: the exact MLE does not converge (near-separation)


def design(X: pd.DataFrame, variety: pd.Series, levels: tuple, *, intercepts: bool, slopes: bool,
           drop_slopes: tuple = ()) -> pd.DataFrame:
    """M0 = cues; M1 += variety indicators; M2 += variety x cue. ``drop_slopes`` cues stay pooled."""
    V = pd.DataFrame({f"v_{lv}": (variety == lv).astype(float) for lv in levels}, index=X.index)
    parts = [X * MAIN_SCALE]
    if intercepts or slopes:
        parts.append(V * MAIN_SCALE)
    if slopes:
        cols = [c for c in X.columns if c not in drop_slopes]
        parts += [X[cols].mul(V[v], axis=0).add_suffix(f":{v}") for v in V]
    return pd.concat(parts, axis=1)


def deviance(D: pd.DataFrame, y) -> float:
    m = LogisticRegression(C=FREE_C, max_iter=5000).fit(D.to_numpy(), y)
    return 2 * len(y) * log_loss(y, m.predict_proba(D.to_numpy()), labels=m.classes_)


def boot_lrt(null: pd.DataFrame, full: pd.DataFrame, y, rng, n_boot: int) -> tuple[float, float]:
    """Deviance drop null -> full and its parametric-bootstrap p-value.

    ``y`` is redrawn from the fitted *null* model, so the reference distribution is
    the true conditional null. (Permuting variety labels would also destroy the
    intercepts and overstate the null, making the slope test conservative.)
    Rows are redrawn independently, which ignores the 55 multi-APC rows' clustering.
    """
    stat = lambda yy: deviance(null, yy) - deviance(full, yy)  # noqa: E731
    obs = stat(y)
    m = LogisticRegression(C=FREE_C, max_iter=5000).fit(null.to_numpy(), y)
    cum = m.predict_proba(null.to_numpy()).cumsum(axis=1)
    draws = [stat(m.classes_[(cum > rng.random((len(cum), 1))).argmax(axis=1)]) for _ in range(n_boot)]
    return obs, (1 + sum(d >= obs for d in draws)) / (1 + n_boot)


def block_weights(model, D: pd.DataFrame, variety: pd.Series, blocks: dict, levels: tuple,
                  *, deviations_only: bool = False) -> pd.DataFrame:
    """Per-variety block weight: mean norm of the block's centred logit contribution.

    The contribution of a block is its columns (pooled slope + that variety's
    deviation) times the fitted coefficients, so it is on the logit scale, does not
    depend on how many columns a block has, and is zero for a cue that never varies.
    Base-rate (intercept) terms are excluded: they are not cue weights.
    ``deviations_only`` keeps just the variety-specific part, i.e. how far the variety's
    weight sits from the pooled one; it is ~0 whenever shrinkage removes the slopes.
    """
    coef, cols = model.coef_, list(D.columns)
    out = {}
    for lv in levels:
        rows = D[(variety == lv).to_numpy()]
        out[lv] = {}
        for b, feats in blocks.items():
            idx = [i for i, c in enumerate(cols) if c.split(":")[0] in feats and (":" in c or not deviations_only)]
            if not idx or rows.empty:
                out[lv][b] = 0.0
                continue
            z = rows.iloc[:, idx].to_numpy() @ coef[:, idx].T
            out[lv][b] = float(np.linalg.norm(z - z.mean(axis=0), axis=1).mean())
    return pd.DataFrame(out)

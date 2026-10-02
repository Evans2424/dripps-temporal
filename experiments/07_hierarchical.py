"""B6 (RQ2): pooled hierarchical model, varying intercepts and varying slopes by variety.

Writes results/tables/b6_{tests,weights,performance}.csv.  Pass --original-only to
drop both later annotation batches (sensitivity run; files get an ``_orig`` suffix).

Nested models over EP/BP/AP/MP (BE excluded: 95% anterior, weights unidentifiable):
  M0 cues | M1 + variety intercepts (base rates) | M2 + variety x cue slopes (weights)
Each test is a deviance drop with a parametric-bootstrap p-value (``hierarchical.boot_lrt``),
Holm-corrected over the six cue blocks.
Slope shrinkage C is chosen by CV *inside* each training fold, so the reported
macro-F1 is nested; the per-variety block weights are conditional on the C chosen
on all the data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from statsmodels.stats.multitest import multipletests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dripps import evaluate, features, io, schema  # noqa: E402
from dripps.hierarchical import block_weights, boot_lrt, design  # noqa: E402

N_BOOT_TEST, N_BOOT_CI = 300, 200
GRID = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
LEVELS = schema.PT_VARIETIES
OUT = ROOT / "results/tables"
K = len(schema.TR_LABELS) - 1  # free parameters per column in a 3-class multinomial


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--original-only", action="store_true", help="drop the later batches; writes *_orig tables")
    orig = ap.parse_args().original_only
    sfx = "_orig" if orig else ""
    df = io.load()
    df = df[df.variety.isin(LEVELS)]
    if orig:
        df = df[df.batch == "original"]
    X, y, g, v = features.build(df), features.target(df), features.groups(df), df.variety
    X = X.loc[:, X.nunique() > 1]  # a cue constant in the pooled data carries no information
    rng = np.random.default_rng(schema.SEED)
    D = lambda **kw: design(X, v, LEVELS, **kw)  # noqa: E731
    print(f"n={len(df)}  varieties={df.variety.value_counts().to_dict()}")

    def test(term, null, full, n_cols):
        obs, p = boot_lrt(null, full, y, rng, N_BOOT_TEST)
        dof = n_cols * (len(LEVELS) - 1) * K
        return term, dof, obs, p

    D0, D1, D2 = D(intercepts=False, slopes=False), D(intercepts=True, slopes=False), D(intercepts=True, slopes=True)
    rows = [test("intercepts (base rates)", D0, D1, 1), test("slopes, all cues", D1, D2, len(X.columns))]
    for b, feats in features.CUE_BLOCKS.items():
        drop = tuple(f for f in feats if f in X.columns)
        if drop:
            rows.append(test(f"slopes, {b}", D(intercepts=True, slopes=True, drop_slopes=drop), D2, len(drop)))
    tests = pd.DataFrame(rows, columns=["term", "df", "dev_drop", "p_boot"])
    blk = tests.term.str.startswith("slopes,") & (tests.term != "slopes, all cues")
    tests.loc[blk, "p_boot_holm"] = multipletests(tests.loc[blk, "p_boot"], method="holm")[1]
    print(tests.round(4).to_string(index=False))
    tests.round(5).to_csv(OUT / f"b6_tests{sfx}.csv", index=False)

    # nested CV: LogisticRegressionCV picks C inside each outer training fold. Its inner folds
    # are shuffled (rows are stored by batch and variety); they ignore sentence groups, which
    # only matters for the few dozen multi-APC rows.
    inner = StratifiedKFold(5, shuffle=True, random_state=schema.SEED)
    cv_model = lambda: LogisticRegressionCV(Cs=list(GRID), cv=inner, scoring="neg_log_loss", max_iter=5000)  # noqa: E731
    perf = []
    for label, M, model in [("M0 cues", D0, LogisticRegression(C=1.0, max_iter=5000)),
                            ("M1 + intercepts", D1, LogisticRegression(C=1.0, max_iter=5000)),
                            ("M2 + slopes", D2, cv_model())]:
        r = evaluate.score(model, M, y, g, label=label)
        perf.append({"model": label, "macro_f1": r["macro_f1"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"], "n": len(df)})
    perf = pd.DataFrame(perf)
    C = float(cv_model().fit(D2.to_numpy(), y).C_[0])
    perf["C_all_data"] = [np.nan, np.nan, C]
    print(perf.round(3).to_string(index=False))
    perf.round(4).to_csv(OUT / f"b6_performance{sfx}.csv", index=False)

    # per-variety block weights from the shrunken model, with a group bootstrap
    def weights(idx):
        m = LogisticRegression(C=C, max_iter=5000).fit(D2.iloc[idx].to_numpy(), y.iloc[idx])
        args = (D2.iloc[idx], v.iloc[idx], features.CUE_BLOCKS, LEVELS)
        return block_weights(m, *args), block_weights(m, *args, deviations_only=True)

    pt = weights(np.arange(len(df)))
    draws = [weights(evaluate.resample_group_rows(g.to_numpy(), rng)) for _ in range(N_BOOT_CI)]
    out = None
    for k, name in enumerate(["weight", "shift"]):  # shift = the variety-specific part of the weight
        d = pd.concat([x[k] for x in draws])
        lo, hi = d.groupby(level=0).quantile(0.025), d.groupby(level=0).quantile(0.975)
        w = pt[k].stack().rename(name).reset_index().rename(columns={"level_0": "block", "level_1": "variety"})
        w[f"{name}_ci_lo"] = [lo.loc[b, vv] for b, vv in zip(w.block, w.variety)]
        w[f"{name}_ci_hi"] = [hi.loc[b, vv] for b, vv in zip(w.block, w.variety)]
        out = w if out is None else out.merge(w, on=["block", "variety"])
    w = out.rename(columns={"weight_ci_lo": "ci_lo", "weight_ci_hi": "ci_hi"})
    w["rank"] = w.groupby("variety")["weight"].rank(ascending=False, method="min").astype(int)
    print(w.pivot(index="block", columns="variety", values="weight").round(3).to_string())
    print("variety-specific shift:\n", w.pivot(index="block", columns="variety", values="shift").round(3).to_string())
    w.round(4).to_csv(OUT / f"b6_weights{sfx}.csv", index=False)


if __name__ == "__main__":
    main()

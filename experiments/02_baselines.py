"""B0-B5 on the four Portuguese varieties, plus the deliberate circular baseline.

Writes results/tables/baselines.csv and b3_by_batch.csv (B3 scored separately on each annotation batch).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dripps import evaluate, features, io, models  # noqa: E402
from dripps.leakage import circular_columns  # noqa: E402


def main() -> None:
    df = io.analysis_frame(io.load())  # PT only; BE is a contrastive reference
    X, y, g = features.build(df), features.target(df), features.groups(df)
    print(f"Portuguese analysis set: {len(df)} rows, {X.shape[1]} features")
    print(f"class balance: {dict(y.value_counts())}\n")

    rows = []
    for label, factory in models.LADDER.items():
        res = evaluate.score(factory(), X, y, g, label=label)
        rows.append({k: v for k, v in res.items() if not k.startswith("_")})
        if label == "B3 logit":
            b3_preds = res["_preds"]
        print(f"  {label:<14} macro-F1 = {res['macro_f1']:.3f} "
              f"[{res['ci_lo']:.3f}, {res['ci_hi']:.3f}]  sd_seed={res['sd_seed']:.4f}")

    # is a batch harder or just different? Out-of-fold B3 predictions, scored per batch.
    yv = y.to_numpy()
    labels = sorted(y.unique())
    by_batch = pd.DataFrame([
        {"batch": b, "n": int(m.sum()), "b3_macro_f1": float(evaluate.macro_f1_per_repeat(yv[m], b3_preds[:, m], labels).mean())}
        for b, m in ((b, (df.batch == b).to_numpy()) for b in df.batch.unique())])
    by_batch.round(4).to_csv(ROOT / "results/tables/b3_by_batch.csv", index=False)
    print("  B3 macro-F1 by batch:", dict(zip(by_batch.batch, by_batch.b3_macro_f1.round(3))))

    # Deliberate upper bound: the definitionally circular annotation columns.
    # The encoder sits inside a Pipeline so it is refit per fold, matching the
    # protocol used by every other rung -- otherwise the comparison is unfair.
    circ = Pipeline([
        ("encode", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
        ("clf", models.b3_logit()),
    ])
    # only rows annotated for SR-SC (the later batches have none), so the bound
    # stays comparable to the original export
    m = (df["SR-SC"] != "").to_numpy()
    X_circ = df[list(circular_columns())][m]
    res = evaluate.score(circ, X_circ, y[m], g[m], label="circular (DR+SR-SC)")
    rows.append({k: v for k, v in res.items() if not k.startswith("_")})
    # same rows, honest cues: the fair comparator for the circular bound, since
    # the ladder above also scores the later batches, which the bound cannot see
    ctrl = evaluate.score(models.b3_logit(), X[m], y[m], g[m], label="circular control: B3 on same rows")
    rows.append({k: v for k, v in ctrl.items() if not k.startswith("_")})
    print(f"\n  {'circ. control':<14} macro-F1 = {ctrl['macro_f1']:.3f} "
          f"[{ctrl['ci_lo']:.3f}, {ctrl['ci_hi']:.3f}]   <- B3 on the {m.sum()} rows the bound can use")
    print(f"\n  {'circular':<14} macro-F1 = {res['macro_f1']:.3f} "
          f"[{res['ci_lo']:.3f}, {res['ci_hi']:.3f}]   <- leakage demo, not a result")

    out = ROOT / "results/tables/baselines.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

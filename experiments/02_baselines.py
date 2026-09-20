"""B0-B5 on the four Portuguese varieties, plus the deliberate circular baseline.

Writes results/tables/baselines.csv.
"""

from __future__ import annotations

import sys
from pathlib import Path

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
        print(f"  {label:<14} macro-F1 = {res['macro_f1']:.3f} "
              f"[{res['ci_lo']:.3f}, {res['ci_hi']:.3f}]  sd_seed={res['sd_seed']:.4f}")

    # Deliberate upper bound: the definitionally circular annotation columns.
    # The encoder sits inside a Pipeline so it is refit per fold, matching the
    # protocol used by every other rung -- otherwise the comparison is unfair.
    circ = Pipeline([
        ("encode", OneHotEncoder(sparse_output=False, handle_unknown="ignore")),
        ("clf", models.b3_logit()),
    ])
    X_circ = df[list(circular_columns())]
    res = evaluate.score(circ, X_circ, y, g, label="circular (DR+SR-SC)")
    rows.append({k: v for k, v in res.items() if not k.startswith("_")})
    print(f"\n  {'circular':<14} macro-F1 = {res['macro_f1']:.3f} "
          f"[{res['ci_lo']:.3f}, {res['ci_hi']:.3f}]   <- leakage demo, not a result")

    out = ROOT / "results/tables/baselines.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

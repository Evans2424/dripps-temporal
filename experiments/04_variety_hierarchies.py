"""RQ2 (first pass): per-variety cue hierarchies.

Descriptive only. Fitting five independent models overfits at n~200 and the
resulting rankings are not strictly comparable; the confirmatory analysis is the
pooled hierarchical model with varying intercepts *and* slopes, which separates
base-rate differences from genuine cue-weight differences. This pass establishes
whether there is a difference worth modelling.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dripps import evaluate, features, interpret, io, models, schema  # noqa: E402


def main() -> None:
    df = io.load()
    rows, floors = {}, {}
    for variety in [*schema.PT_VARIETIES, schema.REFERENCE_VARIETY]:
        sub = df[df.variety == variety]
        X, y, g = features.build(sub), features.target(sub), features.groups(sub)
        # ablation, not permutation: the interaction block is derived
        imp = interpret.block_ablation_importance(
            models.b3_logit(), X, y, g, features.CUE_BLOCKS, n_boot=0
        ).set_index("block")["importance"]  # ranking only, so skip the bootstrap
        rows[variety] = imp
        floors[variety] = {
            "majority": evaluate.score(models.b0_majority(), X, y, g, n_repeats=3)["macro_f1"],
            "logit": evaluate.score(models.b3_logit(), X, y, g, n_repeats=3)["macro_f1"],
            "n": len(sub),
        }

    table = pd.DataFrame(rows).round(4)
    table = table.loc[[b for b in features.CUE_BLOCKS if b in table.index]]
    print("=" * 84)
    print("PER-VARIETY CUE IMPORTANCE (out-of-fold macro-F1 drop when block ablated)")
    print("=" * 84)
    print(table.to_string())

    print("\nRank within each variety (1 = strongest cue):")
    print(table.rank(ascending=False).astype(int).to_string())

    print("\nModel performance per variety:")
    print(pd.DataFrame(floors).T.round(3).to_string())

    table.to_csv(ROOT / "results/tables/variety_cue_importance.csv")
    print(f"\nwrote results/tables/variety_cue_importance.csv")


if __name__ == "__main__":
    main()

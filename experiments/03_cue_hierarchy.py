"""RQ1: the cue hierarchy for the four Portuguese varieties.

Writes results/tables/{cue_importance,logit_coefficients}.csv and
results/tables/tree_rules.txt.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dripps import features, interpret, io, models, schema, stats  # noqa: E402

pd.set_option("display.width", 200)


def main() -> None:
    df = io.analysis_frame(io.load())
    X, y, g = features.build(df), features.target(df), features.groups(df)

    print("=" * 78)
    print("MARGINAL ASSOCIATION (descriptive; confounded by cue collinearity)")
    print("=" * 78)
    marg = pd.DataFrame(
        [stats.association(df, c) for c in schema.CUE_COLUMNS]
    ).sort_values("v", ascending=False)
    print(marg[["cue", "v", "chi2", "dof", "p"]].to_string(index=False,
          float_format=lambda x: f"{x:.3f}"))

    print("\n" + "=" * 78)
    print("SINGLE BEST CUE (B1 one-rule)")
    print("=" * 78)
    oner = models.b1_oner().fit(X, y)
    print(f"  chosen feature : {X.columns[oner.feature_]}")
    print(f"  rules          : {oner.rules_}")
    print(f"  train accuracy : {oner.train_accuracy_:.3f}")

    print("\n" + "=" * 78)
    print("CUE HIERARCHY -- refit ablation (primary): out-of-fold macro-F1 drop")
    print("=" * 78)
    abl = interpret.block_ablation_importance(
        models.b3_logit(), X, y, g, features.CUE_BLOCKS
    )
    print(abl.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\n" + "-" * 78)
    print("cross-check -- block permutation (derived columns recomputed)")
    print("-" * 78)
    perm = interpret.block_permutation_importance(
        models.b3_logit(), X, y, g, features.CUE_BLOCKS
    )
    print(perm.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    imp = abl.merge(perm, on="block", how="outer", suffixes=("_ablation", "_permutation"))
    imp.to_csv(ROOT / "results/tables/cue_importance.csv", index=False)

    print("\n" + "=" * 78)
    print("DIRECTION OF EFFECT -- standardised logit coefficients")
    print("=" * 78)
    pipe = models.b3_logit().fit(X, y)
    coefs = interpret.logit_coefficients(pipe, X.columns, pipe[-1].classes_)
    print(coefs.to_string(float_format=lambda x: f"{x:+.3f}"))
    coefs.to_csv(ROOT / "results/tables/logit_coefficients.csv")

    print("\n" + "=" * 78)
    print("READABLE RULES -- depth-3 tree")
    print("=" * 78)
    tree = models.b4_tree(max_depth=3).fit(X, y)
    rules = interpret.tree_rules(tree, X.columns, tree.classes_)
    print(rules)
    (ROOT / "results/tables/tree_rules.txt").write_text(rules, encoding="utf-8")

    print(f"wrote 3 tables to results/tables/")


if __name__ == "__main__":
    main()

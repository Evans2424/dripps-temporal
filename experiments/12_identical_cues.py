"""Do zero-shot LLMs separate clauses that the annotated cues make identical? (§5.4, descriptive version)

  python experiments/12_identical_cues.py      # writes results/tables/t7_identical_cues.csv

A "cell" is a group of clauses with the same 17-cue vector. B3 sees only the cues, so it cannot rank clauses within a
cell. For each cell with at least two readings and each reading r present in it, we compare the model's raw P(r) on
clauses whose gold reading is r with its P(r) on the other clauses of the cell. Reported: the mean difference, the
share of (cell, reading) pairs where it is positive, and, among clauses the model gets right and B3 gets wrong, how
many are Post/Simul confusions. The confirmatory version (within-cell permutation null, intervals) is test B2 of
docs/T5_ROBUSTNESS_PLAN.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dripps import decision, features, io  # noqa: E402

L = np.array(decision.LABELS)
COLS = ["p_Ant", "p_Post", "p_Simul"]
MODELS = {"jev-1.14": "Jev 1.14", "qwen3.8-27b-fp8": "Qwen3.8-27B"}


def main() -> None:
    df = io.analysis_frame(io.load()).reset_index(drop=True)
    y = features.target(df).to_numpy()
    X = features.build(df)
    cell = pd.Series([hash(tuple(r)) for r in X.to_numpy()])
    t5 = pd.read_csv(ROOT / "results/tables/t5_decision_preds.csv")
    rows = []
    for key, label in MODELS.items():
        P = decision.debias(np.load(ROOT / f"results/cache_decision/{key}/plain.npy"))  # raw, zero-shot
        fused = t5[t5.variant == f"B3 + decision {key} calibrated, w=0.5"].pivot_table(index="id", values=COLS)
        cal = t5[t5.variant == f"decision {key} calibrated"].pivot_table(index="id", values=COLS)
        b3 = (2 * fused - cal).loc[df.ID][COLS].to_numpy()  # B3's own probabilities, averaged over seeds
        pm, pb = L[P.argmax(-1)], L[b3.argmax(-1)]

        gaps = []
        for c in cell.unique():
            ix = np.where((cell == c).to_numpy())[0]
            if len(set(y[ix])) < 2:
                continue
            for r in set(y[ix]):
                k = list(L).index(r)
                gaps.append(P[ix[y[ix] == r], k].mean() - P[ix[y[ix] != r], k].mean())
        gaps = np.array(gaps)
        mixed = cell.map(pd.DataFrame({"c": cell, "y": y}).groupby("c")["y"].nunique()) > 1
        wins = (pm == y) & (pb != y)
        ps = wins & np.isin(y, ["Post", "Simul"]) & np.isin(pb, ["Post", "Simul"])
        rows.append({"model": label, "clauses_in_mixed_cells": int(mixed.sum()),
                     "mixed_cells": int(cell[mixed.to_numpy()].nunique()), "cell_reading_pairs": len(gaps),
                     "mean_gap": float(gaps.mean()), "share_positive": float((gaps > 0).mean()),
                     "model_right_b3_wrong": int(wins.sum()), "of_which_post_simul": int(ps.sum()),
                     "agreement_with_b3": float((pm == pb).mean())})
        print(rows[-1], flush=True)
    pd.DataFrame(rows).round(4).to_csv(ROOT / "results/tables/t7_identical_cues.csv", index=False)


if __name__ == "__main__":
    main()

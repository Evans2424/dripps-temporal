"""The RQ2 paired contrasts recover a planted matching advantage and find none when there is none."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("v13", Path(__file__).resolve().parents[1] / "experiments/13_variety_models.py")
v13 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v13)


def frames(acc_a: dict, acc_b: dict, n=120, seeds=2, rng=None):
    """Two models' predictions on the same clauses; accuracy per variety as given (errors pick another label)."""
    labs = np.array(["Ant", "Post", "Simul"])
    rows = {"a": [], "b": []}
    for v in ("EP", "BP", "AP", "MP"):
        true = rng.choice(labs, size=n)
        for s in range(seeds):
            for key, acc in (("a", acc_a), ("b", acc_b)):
                ok = rng.random(n) < acc[v]
                pred = np.where(ok, true, [rng.choice(labs[labs != t]) for t in true])
                rows[key].append(pd.DataFrame({"id": [f"{v}{i}" for i in range(n)], "seed": s, "true": true,
                                               "pred": pred, "variety": v, "grp": [f"{v}{i // 2}" for i in range(n)]}))
    return [pd.concat(rows[k], ignore_index=True) for k in ("a", "b")]


def test_a_planted_matching_advantage_is_found():
    rng = np.random.default_rng(0)
    a, b = frames({"EP": .85, "BP": .55, "AP": .8, "MP": .8}, {"EP": .55, "BP": .85, "AP": .5, "MP": .5}, rng=rng)
    est, boots = v13.paired_boot(a, b, rng, n_boot=200)
    assert est["H1"] > 0.4 and boots.H1.quantile(0.025) > 0
    assert est["H2"] > 0.2 and boots.H2.quantile(0.025) > 0


def test_identical_models_give_no_contrast():
    rng = np.random.default_rng(1)
    a, _ = frames({v: .7 for v in ("EP", "BP", "AP", "MP")}, {v: .7 for v in ("EP", "BP", "AP", "MP")}, rng=rng)
    est, boots = v13.paired_boot(a, a.copy(), rng, n_boot=50)
    assert est == {"H1": 0.0, "H2": 0.0} and (boots == 0).all().all()


def test_holm_is_monotone_and_capped():
    assert v13.holm([0.01, 0.04, 0.03, 0.5]) == [0.04, 0.09, 0.09, 0.5]

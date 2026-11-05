"""RQ2 confirmatory tests: does a model score higher on the variety it was trained on? (docs/VARIETY_MODELS_PLAN.md)

  python experiments/13_variety_models.py     # writes results/tables/t8_variety_{contrasts,gate,nll}.csv

For each PT-PT / PT-BR pair that differs only in training variety (Gervásio-7B, Albertina-100m), on the same clauses:
  H1  DiD = [F1(PT-PT, EP) - F1(PT-BR, EP)] - [F1(PT-PT, BP) - F1(PT-BR, BP)] > 0   (matching advantage)
  H2  F1(PT-PT, AP+MP) - F1(PT-BR, AP+MP) > 0                                      (African transfer)
Macro-F1 averaged over seeds; a paired bootstrap resamples sentence groups within each variety, with the same draws
for both models; one-sided p = share of draws <= 0, Holm-corrected over every test that ran. Gate: a model enters
only if its pooled macro-F1 interval excludes 0.33 (uniform guessing). M1: mean token NLL of the sentences by variety
(manipulation check: is the training variety visible to the model at all?). The per-variety table of every model is
t6_variety_by_model.csv (11_variety_breakdown.py).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dripps import io  # noqa: E402
from dripps.schema import SEED  # noqa: E402

N_BOOT = 1000
CHANCE = 1 / 3
#: name -> (PT-PT variant, PT-BR variant, predictions file)
PAIRS = {"Gervásio-7B": ("decision gervasio-7b-ptpt-t2 calibrated", "decision gervasio-7b-ptbr-t2 calibrated", "t5"),
         "Albertina-100m": ("marked@albertina", "marked@albertina-ptbr", "t1")}
#: LLM caches whose sentence NLL feeds M1, as PT-PT / PT-BR pairs
NLL_PAIRS = {"Gervásio-7B": ("gervasio-7b-ptpt-t2", "gervasio-7b-ptbr-t2")}


def f1(d: pd.DataFrame) -> float:
    return float(np.mean([f1_score(s.true, s.pred, average="macro") for _, s in d.groupby("seed")]))


def stats(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """The two contrasts for prediction frames a (PT-PT) and b (PT-BR) with a ``variety`` column."""
    def gap(vs):
        return f1(a[a.variety.isin(vs)]) - f1(b[b.variety.isin(vs)])
    return {"H1": gap(["EP"]) - gap(["BP"]), "H2": gap(["AP", "MP"])}


def paired_boot(a: pd.DataFrame, b: pd.DataFrame, rng, n_boot: int = N_BOOT):
    """Point estimates and bootstrap draws of ``stats``: sentence groups resampled within each variety, the same draw
    applied to both models (a and b need ``id``, ``variety`` and ``grp`` columns over the same clauses)."""
    ga, gb = a.groupby("grp").indices, b.groupby("grp").indices
    by_var = a.drop_duplicates("grp").groupby("variety").grp.apply(list).to_dict()
    draws = []
    for _ in range(n_boot):
        keys = [k for ks in by_var.values() for k in rng.choice(ks, size=len(ks))]
        draws.append(stats(a.iloc[np.concatenate([ga[k] for k in keys])], b.iloc[np.concatenate([gb[k] for k in keys])]))
    return stats(a, b), pd.DataFrame(draws)


def holm(p: list[float]) -> list[float]:
    order, out, run = np.argsort(p), np.empty(len(p)), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(p) - r) * p[i]))
        out[i] = run
    return out.tolist()


def main() -> None:
    df = io.analysis_frame(io.load())
    variety, group = df.set_index("ID").variety, df.set_index("ID").sentence_group
    preds = {"t1": pd.read_csv(ROOT / "results/tables/t1_encoder_preds.csv"),
             "t5": pd.read_csv(ROOT / "results/tables/t5_decision_preds.csv")}
    for p in preds.values():
        p["variety"], p["grp"] = variety.loc[p.id].to_numpy(), group.loc[p.id].to_numpy()
    rng = np.random.default_rng(SEED)

    gate = []
    for t, p in preds.items():
        for v in sorted(p.variant.unique()):
            in_pair = v in {x[i] for x in PAIRS.values() for i in (0, 1)}
            if not (in_pair or (v.startswith("decision ") and v.endswith("-t2 calibrated"))):  # the RQ2 grid only
                continue
            d = p[p.variant == v].reset_index(drop=True)
            idx, keys = d.groupby("grp").indices, d.grp.unique()
            boots = [f1(d.iloc[np.concatenate([idx[k] for k in rng.choice(keys, size=len(keys))])]) for _ in range(N_BOOT)]
            lo, hi = np.percentile(boots, [2.5, 97.5])
            gate.append({"variant": v, "macro_f1": f1(d), "ci_lo": lo, "ci_hi": hi, "passes": bool(lo > CHANCE)})
    gate = pd.DataFrame(gate)
    passes = dict(zip(gate.variant, gate.passes))

    rows = []
    for pair, (va, vb, t) in PAIRS.items():
        p = preds[t]
        a, b = (p[p.variant == v].sort_values(["seed", "id"]).reset_index(drop=True) for v in (va, vb))
        if a.empty or b.empty:
            print(f"{pair}: not run yet ({va if a.empty else vb} missing)")
            continue
        if not (passes[va] and passes[vb]):
            print(f"{pair}: excluded by the gate (pooled macro-F1 interval does not exclude {CHANCE:.2f})")
            continue
        est, boots = paired_boot(a, b, rng)
        for h in ("H1", "H2"):
            rows.append({"pair": pair, "test": h, "estimate": est[h], "ci_lo": boots[h].quantile(0.025),
                         "ci_hi": boots[h].quantile(0.975), "p_one_sided": (1 + (boots[h] <= 0).sum()) / (1 + N_BOOT)})
    out = pd.DataFrame(rows, columns=["pair", "test", "estimate", "ci_lo", "ci_hi", "p_one_sided"])
    if len(out):
        out["p_holm"] = holm(out.p_one_sided.tolist())
        out["within_0.05"] = (out.ci_lo > -0.05) & (out.ci_hi < 0.05)

    nll = []
    for pair, names in NLL_PAIRS.items():
        paths = [ROOT / "results/cache_decision" / n / "nll.npy" for n in names]
        if not all(x.exists() for x in paths):
            continue
        for n, x in zip(names, paths):
            s = pd.Series(np.load(x), index=df.ID)
            for v, m in s.groupby(variety.loc[s.index].to_numpy()).mean().items():
                nll.append({"pair": pair, "model": n, "variety": v, "mean_nll": m})

    tab = ROOT / "results/tables"
    gate.round(4).to_csv(tab / "t8_variety_gate.csv", index=False)
    out.round(4).to_csv(tab / "t8_variety_contrasts.csv", index=False)
    pd.DataFrame(nll).round(4).to_csv(tab / "t8_variety_nll.csv", index=False)
    print(out.round(3).to_string(index=False) if len(out) else "no pair complete yet")


if __name__ == "__main__":
    main()

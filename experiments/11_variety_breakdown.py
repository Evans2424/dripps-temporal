"""Per-variety macro-F1 of the models already run (RQ2, descriptive): out-of-fold predictions split by variety.

  python experiments/11_variety_breakdown.py        # writes results/tables/t6_variety_by_model.csv

Reads the saved predictions (t1_encoder_preds.csv, t5_decision_preds.csv); B3's own per-clause probabilities are not
stored, so they are recovered exactly as 2 * (B3 + model, w=0.5) - model, averaged over seeds. Macro-F1 per variety is
averaged over the 5 seeds; the 95% interval resamples sentence groups within the variety (1,000 draws).
Descriptive only: per-variety n is 193-345, and the model x variety interaction is the confirmatory test (planned).
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

VARIETIES = ["EP", "BP", "AP", "MP"]
COLS = ["p_Ant", "p_Post", "p_Simul"]
N_BOOT = 1000


def b3_predictions(t5: pd.DataFrame) -> pd.DataFrame:
    fused, model = "B3 + decision qwen3.8-27b-fp8 calibrated, w=0.5", "decision qwen3.8-27b-fp8 calibrated"
    f = t5[t5.variant == fused].set_index(["id", "seed"])
    m = t5[t5.variant == model].set_index(["id", "seed"])
    p = 2 * f[COLS] - m[COLS]
    out = p.reset_index()
    out["pred"] = np.array(["Ant", "Post", "Simul"])[p.to_numpy().argmax(1)]
    out["true"] = m.loc[p.index, "true"].to_numpy()
    return out


def macro_f1(x: pd.DataFrame) -> float:
    return float(np.mean([f1_score(s.true, s.pred, average="macro") for _, s in x.groupby("seed")]))


def main() -> None:
    df = io.analysis_frame(io.load())
    variety, group = df.set_index("ID").variety, df.set_index("ID").sentence_group
    t1 = pd.read_csv(ROOT / "results/tables/t1_encoder_preds.csv")
    t5 = pd.read_csv(ROOT / "results/tables/t5_decision_preds.csv")
    models = {
        "B3 logit (annotated cues)": b3_predictions(t5),
        "encoder marked (xlm-roberta-base)": t1[t1.variant == "marked"],
        "encoder marked@albertina (PT-PT)": t1[t1.variant == "marked@albertina"],
        "encoder marked@bertimbau (PT-BR)": t1[t1.variant == "marked@bertimbau"],
        "Qwen3.8-27B raw": t5[t5.variant == "decision qwen3.8-27b-fp8"],
        "Qwen3.8-27B calibrated": t5[t5.variant == "decision qwen3.8-27b-fp8 calibrated"],
        "Jev 1.14 raw": t5[t5.variant == "decision jev-1.14"],
        "B3 + Qwen3.8-27B calibrated": t5[t5.variant == "B3 + decision qwen3.8-27b-fp8 calibrated, w=0.5"],
    }
    # RQ2 grid (docs/VARIETY_MODELS_PLAN.md): every template-2 model that has run, and the PT-BR Albertina twin
    for v in sorted(t5.variant.unique()):
        if v.startswith("decision ") and v.endswith("-t2 calibrated"):
            models[v.removeprefix("decision ").replace("-t2 calibrated", " calibrated (template 2)")] = t5[t5.variant == v]
    if (t1.variant == "marked@albertina-ptbr").any():
        models["encoder marked@albertina-ptbr (PT-BR)"] = t1[t1.variant == "marked@albertina-ptbr"]
    rng = np.random.default_rng(SEED)
    rows = []
    for name, d in models.items():
        d = d.copy()
        d["variety"], d["grp"] = variety.loc[d.id].to_numpy(), group.loc[d.id].to_numpy()
        for v in VARIETIES:
            x = d[d.variety == v].reset_index(drop=True)
            idx = {k: ix.to_numpy() for k, ix in x.groupby("grp").groups.items()}
            keys = list(idx)
            boots = [macro_f1(x.loc[np.concatenate([idx[keys[i]] for i in rng.integers(len(keys), size=len(keys))])])
                     for _ in range(N_BOOT)]
            rows.append({"model": name, "variety": v, "n": int(x.id.nunique()), "macro_f1": macro_f1(x),
                         "ci_lo": float(np.percentile(boots, 2.5)), "ci_hi": float(np.percentile(boots, 97.5))})
            print(f"{name:36} {v}  {rows[-1]['macro_f1']:.3f} [{rows[-1]['ci_lo']:.3f}, {rows[-1]['ci_hi']:.3f}]", flush=True)
    pd.DataFrame(rows).round(4).to_csv(ROOT / "results/tables/t6_variety_by_model.csv", index=False)


if __name__ == "__main__":
    main()

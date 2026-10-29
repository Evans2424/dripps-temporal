"""Merge the tables of a later 08_encoder.py run (another cache, e.g. the lab's round 3) into results/tables.

  python experiments/09_merge_encoder.py results_bertimbau/tables

Rows, predictions and chosen configs of variants the run computed are added next to the earlier ones.
The paired contrasts that need both sets (a variant against its cue-free twin and against the same
variant on xlm-roberta-base) are computed here from the saved predictions with 08_encoder.py's own
bootstrap, so they match what a single run with every cache present would write. t3 is untouched.
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dripps import encoder, features, io  # noqa: E402

spec = importlib.util.spec_from_file_location("enc08", ROOT / "experiments/08_encoder.py")
e08 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e08)

if len(sys.argv) != 2:
    sys.exit("usage: python experiments/09_merge_encoder.py <tables dir of the later run>")
tab, src = ROOT / "results/tables", Path(sys.argv[1])


def merged(name, key):
    old, new = pd.read_csv(tab / name), pd.read_csv(src / name)
    return pd.concat([old[~old[key].isin(new[key])], new], ignore_index=True)


preds = merged("t1_encoder_preds.csv", "variant")
config = merged("t1_encoder_config.csv", "variant")
old, new = pd.read_csv(tab / "t1_encoder.csv"), pd.read_csv(src / "t1_encoder.csv")
# a variant the new run recomputed replaces every old row that mentions it (summary, fusion, contrasts)
redo = re.compile("encoder (?:" + "|".join(re.escape(v) for v in pd.read_csv(src / "t1_encoder_preds.csv").variant.unique()) + r")(?![\w@])")
old = old[~old.model.str.contains(redo) & ~old.model.isin(new.model)]
rows = pd.concat([old, new], ignore_index=True)

df = io.analysis_frame(io.load())
y, g = features.target(df).to_numpy(), features.groups(df).to_numpy()
labels = sorted(pd.unique(y))
pos = pd.Series(np.arange(len(df)), index=df.ID.to_numpy())
guess = {}
for var, d in preds.groupby("variant"):
    a = np.empty((d.seed.nunique(), len(df)), object)
    a[d.seed.to_numpy(), pos[d.id].to_numpy()] = d.pred.to_numpy()
    guess[f"encoder {var}"] = a
n_seeds = next(iter(guess.values())).shape[0]

pairs = []
for v in preds.variant.unique():  # the same twins as 08_encoder.py
    b, m, c = encoder.parse(v)
    for twin in (encoder.name(b, "", c) if m else None, encoder.name(b, m, "xlmr") if c != "xlmr" else None,
                 encoder.name("plain", m, c) if b == "marked" else None):
        if twin and f"encoder {twin}" in guess and (f"encoder {v}", f"encoder {twin}") not in pairs:
            pairs.append((f"encoder {v}", f"encoder {twin}"))
have = set(rows.model)
amb = (df.groupby("sentence_norm")[features.target(df).name].transform("nunique") > 1).to_numpy()
extra = []
for tag, m in (("", np.ones(len(y), bool)), (" (text-unambiguous rows)", ~amb)):
    draws = e08.group_draws(g[m])
    rng = np.random.default_rng(e08.SEED)
    rs = [rng.integers(n_seeds) for _ in draws]
    boots = {k: e08.boot_f1(y[m], v[:, m], labels, draws, rs) for k, v in guess.items()}
    for a, b in pairs:
        name = f"{a} - {b} (paired){tag}"
        if name not in have:
            d, lo, hi = e08.paired(y[m], guess[a][:, m], guess[b][:, m], labels, (boots[a], boots[b]))
            extra.append({"model": name, "n": int(m.sum()), "macro_f1": d, "ci_lo": lo, "ci_hi": hi})
out = {"t1_encoder.csv": pd.concat([rows, pd.DataFrame(extra)], ignore_index=True).round(4),
       "t1_encoder_preds.csv": preds.round(5), "t1_encoder_config.csv": config}
for name, t in out.items():  # write all three aside first, then swap, so a failure never leaves them out of step
    t.to_csv(tab / f"{name}.tmp", index=False)
for name in out:
    os.replace(tab / f"{name}.tmp", tab / name)
print(f"{len(extra)} new paired contrasts; {preds.variant.nunique()} variants in the predictions")

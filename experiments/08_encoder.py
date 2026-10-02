"""T1 + T3: fine-tuned xlm-roberta-base on the whole sentence, then causal ablation.

Needs a GPU and requirements-neural.txt (run on the lab machine, copy the tables back).
Every perturbed text is scored by the fold's own model on its held-out rows, so
the ablation is out-of-fold and no checkpoint is kept. Folds are B3's folds.

Writes results/tables/t1_encoder.csv, t1_encoder_preds.csv, t3_ablation.csv and
t3_span_sample.csv (50 rows to check span recovery by hand).

  --smoke   100 rows, one seed, one epoch, writes nothing (catches shape/OOM bugs)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dripps import ablate, encoder, evaluate, features, io, models  # noqa: E402
from dripps.schema import SEED  # noqa: E402

N_SEEDS = 5  # the plan's minimum; B3 is rescored on the same seeds
VARIANTS = ["mask_participle", "mask_main_verb", "mask_adverbials", "move_apc_initial",
            "ctrl_participle", "ctrl_main_verb", "ctrl_adverbials"]


def build_variants(texts, mask):
    import spacy
    nlp = spacy.load("pt_core_news_sm", disable=["ner", "lemmatizer"])
    rng = np.random.default_rng(SEED)
    out = {v: [None] * len(texts) for v in VARIANTS}
    rows = []
    for i, doc in enumerate(nlp.pipe(texts, batch_size=64)):
        t = texts[i]
        apc = ablate.apc_span(t)
        verb = ablate.main_verb_span(doc, apc) if apc else None
        adv = ablate.adverbial_spans(t)
        # a control may not remove the APC itself, the verb it is compared with, or (for the adverbial
        # control) the adverbials; masking "tendo" anywhere would change the construction under test
        keep = ablate.tendo_spans(t) + ([apc[0]] if apc else []) + ([verb] if verb else [])
        if apc:
            out["mask_participle"][i] = ablate.mask_spans(t, [apc[1]], mask)
            out["ctrl_participle"][i] = ablate.mask_random(t, keep, 1, mask, rng)
        if verb:
            out["mask_main_verb"][i] = ablate.mask_spans(t, [verb], mask)
            out["ctrl_main_verb"][i] = ablate.mask_random(t, keep, 1, mask, rng)
        if adv:
            out["mask_adverbials"][i] = ablate.mask_spans(t, adv, mask)
            out["ctrl_adverbials"][i] = ablate.mask_random(t, ablate.tendo_spans(t) + ([apc[1]] if apc else []) + ([verb] if verb else []) + adv, len(adv), mask, rng)
        first = doc[0]
        lower = first.pos_ not in ("PROPN", "X", "NUM") and not (len(first) > 1 and first.is_upper)
        out["move_apc_initial"][i] = ablate.move_apc_initial(t, lower_first=lower)
        rows.append({"i": i, "sentence": t,
                     "participle": t[slice(*apc[1])] if apc else "",
                     "main_verb": t[slice(*verb)] if verb else "",
                     "adverbials": " | ".join(t[s:e] for s, e in adv)})
    return out, pd.DataFrame(rows)


def group_draws(groups, n_boot=2000):
    """Row indices of ``n_boot`` sentence-group bootstrap draws; built once per row subset."""
    rng = np.random.default_rng(SEED)
    return [evaluate.resample_group_rows(groups, rng) for _ in range(n_boot)]


def mean_ci(values, draws):
    """Mean of per-row ``values`` and its group-bootstrap 95% CI."""
    boot = [values[idx].mean() for idx in draws]
    return float(values.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def paired(y, a, b, labels, draws):
    """Observed macro-F1(a) - macro-F1(b) over (seeds, n) predictions, with a group-bootstrap CI."""
    obs = evaluate.macro_f1_per_repeat(y, a, labels).mean() - evaluate.macro_f1_per_repeat(y, b, labels).mean()
    rng = np.random.default_rng(SEED)
    boot = []
    for idx in draws:
        r = rng.integers(len(a))  # fold the seed jitter in, as evaluate.bootstrap_group_ci does
        boot.append(evaluate.macro_f1(y[idx], a[r][idx], labels) - evaluate.macro_f1(y[idx], b[r][idx], labels))
    return float(obs), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    smoke = ap.parse_args().smoke

    import torch
    if not smoke and not torch.cuda.is_available():
        sys.exit("no CUDA device: 25 XLM-R fine-tunes on CPU would take days. Run this on the GPU machine.")
    from transformers import AutoTokenizer
    mask = AutoTokenizer.from_pretrained(encoder.CHECKPOINT).mask_token

    df = io.analysis_frame(io.load())
    if smoke:
        df = df.sample(100, random_state=SEED).reset_index(drop=True)
    texts = df.sentence_norm.tolist()
    y = features.target(df).to_numpy()
    g = features.groups(df).to_numpy()
    labels = sorted(pd.unique(y))
    n_seeds = 1 if smoke else N_SEEDS
    hp = dict(epochs=1) if smoke else {}
    print(f"{len(df)} rows, {n_seeds} seeds, checkpoint {encoder.CHECKPOINT}", flush=True)

    variants, span_df = build_variants(texts, mask)
    print({v: sum(t is not None for t in variants[v]) for v in VARIANTS}, flush=True)

    probs = np.full((n_seeds, len(y), len(labels)), np.nan)
    vprobs = {v: np.full_like(probs, np.nan) for v in VARIANTS}
    for r in range(n_seeds):
        for k, (train, test) in enumerate(evaluate.folds(y, g, SEED + r)):
            ev = {"orig": [texts[i] for i in test]}
            where = {v: [i for i in test if variants[v][i] is not None] for v in VARIANTS}
            ev.update({v: [variants[v][i] for i in where[v]] for v in VARIANTS})
            out = encoder.fit_predict([texts[i] for i in train], y[train], ev,
                                      labels=labels, seed=SEED + r, **hp)
            probs[r, test] = out["orig"]
            for v in VARIANTS:
                if where[v]:
                    vprobs[v][r, where[v]] = out[v]
            print(f"seed {r} fold {k} done", flush=True)
    if smoke:
        print("smoke ok; probabilities finite:", bool(np.isfinite(probs).all()))
        return

    # ---- T1: encoder vs B3 on the same seeds and folds ----
    lab = np.array(labels)
    enc = lab[probs.argmax(-1)]
    b3 = evaluate.cv_predict(models.b3_logit(), features.build(df), y, g, n_repeats=n_seeds)
    # Sentences annotated for several APCs appear as identical text with different labels. A text-only
    # model cannot separate them (B3's per-clause features can), so score the other rows on their own.
    amb = (df.groupby("sentence_norm")[features.target(df).name].transform("nunique") > 1).to_numpy()
    rows = []
    for tag, m in (("", np.ones(len(y), bool)), (" (text-unambiguous rows)", ~amb)):
        for name, p in (("B3 logit", b3), (f"encoder {encoder.CHECKPOINT}", enc)):
            ym, pm, gm = y[m], p[:, m], g[m]
            per = evaluate.macro_f1_per_repeat(ym, pm, labels)
            lo, hi = evaluate.bootstrap_group_ci(ym, pm, gm)
            f1c = np.mean([f1_score(ym, pr, labels=labels, average=None, zero_division=0) for pr in pm], axis=0)
            rows.append({"model": name + tag, "n": int(m.sum()), "macro_f1": per.mean(), "ci_lo": lo, "ci_hi": hi,
                         "sd_seed": per.std(ddof=1), **{f"f1_{c}": v for c, v in zip(labels, f1c)}})
        d, dlo, dhi = paired(y[m], enc[:, m], b3[:, m], labels, group_draws(g[m]))
        rows.append({"model": f"encoder - B3 (paired){tag}", "n": int(m.sum()), "macro_f1": d, "ci_lo": dlo, "ci_hi": dhi})
    tab = ROOT / "results/tables"
    pd.DataFrame(rows).round(4).to_csv(tab / "t1_encoder.csv", index=False)

    pred_rows = pd.DataFrame({"id": np.tile(df.ID.to_numpy(), n_seeds), "seed": np.repeat(np.arange(n_seeds), len(y)),
                              "true": np.tile(y, n_seeds), "pred": enc.ravel(),
                              **{f"p_{c}": probs[:, :, j].ravel() for j, c in enumerate(labels)}})
    pred_rows.round(5).to_csv(tab / "t1_encoder_preds.csv", index=False)

    # ---- T3: change in the predicted distribution under each perturbation ----
    true_j = np.array([labels.index(v) for v in y])
    p_true = lambda P: np.take_along_axis(P, true_j[None, :, None], 2)[..., 0]  # (seeds, n)
    base = probs.mean(0)
    rows, d_true = [], {}
    for v in VARIANTS:
        ok = ~np.isnan(vprobs[v][0, :, 0])
        delta = vprobs[v].mean(0)[ok] - base[ok]                       # (m, K)
        dt = (p_true(vprobs[v]).mean(0) - p_true(probs).mean(0))       # (n,), NaN where n/a
        d_true[v] = dt
        draws = group_draws(g[ok])
        row = {"perturbation": v, "n_rows": int(ok.sum())}
        for j, c in enumerate(labels):
            m, lo, hi = mean_ci(delta[:, j], draws)
            row.update({f"d_P_{c}": m, f"d_P_{c}_lo": lo, f"d_P_{c}_hi": hi})
        row["d_P_true"], row["d_P_true_lo"], row["d_P_true_hi"] = mean_ci(dt[ok], draws)
        rows.append(row)
    both = ~np.isnan(d_true["mask_participle"]) & ~np.isnan(d_true["mask_main_verb"])
    diff = (d_true["mask_main_verb"] - d_true["mask_participle"])[both]
    m, lo, hi = mean_ci(diff, group_draws(g[both]))
    rows.append({"perturbation": "main_verb - participle (d_P_true)", "n_rows": int(both.sum()),
                 "d_P_true": m, "d_P_true_lo": lo, "d_P_true_hi": hi})
    pd.DataFrame(rows).round(4).to_csv(tab / "t3_ablation.csv", index=False)

    span_df.insert(1, "ID", df.ID.to_numpy()[span_df.i])
    # hand-check rows where a span was found: the failure modes are wrong spans, not missing ones
    found = span_df[(span_df.participle != "") & (span_df.main_verb != "")]
    sample = found.sample(50, random_state=SEED).assign(participle_ok="", main_verb_ok="")
    print(f"span recovery: participle {int((span_df.participle != '').sum())}, "
          f"main verb {int((span_df.main_verb != '').sum())}, both {len(found)} of {len(span_df)}")
    sample.to_csv(tab / "t3_span_sample.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string())


if __name__ == "__main__":
    main()

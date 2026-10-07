"""T1 + T3: tuned xlm-roberta-base on the sentence, plain and with the two verbs marked.

Needs a GPU and requirements-neural.txt (run on the lab machine, copy the tables back).
Hyperparameters are chosen inside each training fold (encoder.select_and_predict), folds are
B3's folds, and each (variant, seed, fold) is cached in results/cache_encoder/ so a crash in a
multi-hour run loses one fold. T3 perturbations are scored by the plain model's fold model, so
the ablation is out-of-fold and no checkpoint is kept. T1 also fuses B3 and encoder
probabilities (fixed average, nothing tuned) to ask whether text adds to the inventory.

Writes results/tables/t1_encoder.csv, t1_encoder_preds.csv, t1_encoder_config.csv,
t3_ablation.csv and t3_span_sample.csv (50 rows to check span recovery by hand).

  --variant plain|marked|both   which variants to compute (cached ones are reused)
  --smoke                       100 rows, one seed, one config, one epoch, writes nothing
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
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
    rows, marked, marks = [], [], []
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
        wraps = [(sp, o, c) for sp, o, c in ((apc[0] if apc else None, "[[ ", " ]]"), (verb, "<< ", " >>")) if sp]
        mt, shift = ablate.mark(t, wraps)
        marked.append(mt)
        marks.append((shift(apc[1][0]) if apc else None, shift(verb[0]) if verb else None))
        rows.append({"i": i, "sentence": t,
                     "participle": t[slice(*apc[1])] if apc else "",
                     "main_verb": t[slice(*verb)] if verb else "",
                     "adverbials": " | ".join(t[s:e] for s, e in adv)})
    return out, pd.DataFrame(rows), marked, marks


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


def summarise(name, p, y, g, labels, m):
    ym, pm, gm = y[m], p[:, m], g[m]
    per = evaluate.macro_f1_per_repeat(ym, pm, labels)
    lo, hi = evaluate.bootstrap_group_ci(ym, pm, gm)
    f1c = np.mean([f1_score(ym, pr, labels=labels, average=None, zero_division=0) for pr in pm], axis=0)
    return {"model": name, "n": int(m.sum()), "macro_f1": per.mean(), "ci_lo": lo, "ci_hi": hi,
            "sd_seed": per.std(ddof=1), **{f"f1_{c}": v for c, v in zip(labels, f1c)}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--variant", choices=["plain", "marked", "both"], default="both")
    args = ap.parse_args()
    smoke = args.smoke
    todo = ["plain", "marked"] if args.variant == "both" else [args.variant]

    import torch
    if not smoke and not torch.cuda.is_available():
        sys.exit("no CUDA device: the tuned runs would take days on CPU. Run this on the GPU machine.")
    from transformers import AutoTokenizer
    mask = AutoTokenizer.from_pretrained(encoder.CHECKPOINT).mask_token

    df = io.analysis_frame(io.load())
    if smoke:
        df = df.sample(100, random_state=SEED).reset_index(drop=True)
    texts = df.sentence_norm.tolist()
    y = features.target(df).to_numpy()
    g = features.groups(df).to_numpy()
    X = features.build(df).to_numpy()
    labels = sorted(pd.unique(y))
    n_seeds = 1 if smoke else N_SEEDS
    configs = [dict(lr=3e-5, epochs=1)] if smoke else encoder.CONFIGS
    cache = ROOT / "results/cache_encoder"
    cache.mkdir(parents=True, exist_ok=True)
    print(f"{len(df)} rows, {n_seeds} seeds, {len(configs)} configs, variants {todo}", flush=True)

    perturbed, span_df, marked, marks = build_variants(texts, mask)
    print({v: sum(t is not None for t in perturbed[v]) for v in VARIANTS},
          "| marks:", sum(m[0] is not None for m in marks), "participle,",
          sum(m[1] is not None for m in marks), "main verb", flush=True)

    probs = {v: np.full((n_seeds, len(y), len(labels)), np.nan) for v in todo}
    b3p = np.full((n_seeds, len(y), len(labels)), np.nan)
    vprobs = {v: np.full_like(probs["plain"], np.nan) for v in VARIANTS} if "plain" in todo else {}
    chosen = []
    for r in range(n_seeds):
        for k, (train, test) in enumerate(evaluate.folds(y, g, SEED + r)):
            b3p[r, test] = clone(models.b3_logit()).fit(X[train], y[train]).predict_proba(X[test])
            for var in todo:
                f = cache / f"{var}_s{r}_f{k}.npz"
                if f.exists() and not smoke:
                    z = np.load(f)
                else:
                    if var == "plain":
                        where = {v: [i for i in test if perturbed[v][i] is not None] for v in VARIANTS}
                        ev = {"orig": [texts[i] for i in test],
                              **{v: [perturbed[v][i] for i in where[v]] for v in VARIANTS}}
                        kw = {}
                    else:
                        ev, kw = {"orig": [marked[i] for i in test]}, dict(
                            train_marks=[marks[i] for i in train], eval_marks={"orig": [marks[i] for i in test]})
                    src = texts if var == "plain" else marked
                    out, cfg = encoder.select_and_predict(
                        [src[i] for i in train], y[train], g[train], ev, labels=labels, seed=SEED + r,
                        configs=configs, **kw)
                    flat = {"orig": out["orig"], "cfg": np.array(json.dumps(cfg))}
                    if var == "plain":
                        for v in VARIANTS:
                            flat[f"i_{v}"], flat[f"p_{v}"] = np.array(where[v], dtype=int), out[v]
                    if not smoke:
                        np.savez(f, **flat)
                    z = flat
                probs[var][r, test] = z["orig"]
                chosen.append({"variant": var, "seed": r, "fold": k, **json.loads(str(z["cfg"]))})
                if var == "plain":
                    for v in VARIANTS:
                        if len(z[f"i_{v}"]):
                            vprobs[v][r, z[f"i_{v}"]] = z[f"p_{v}"]
            print(f"seed {r} fold {k} done", flush=True)
    if smoke:
        print("smoke ok; probabilities finite:", all(bool(np.isfinite(p).all()) for p in probs.values()))
        return

    # ---- T1: encoders and fusion vs B3, same seeds and folds ----
    lab = np.array(labels)
    guess = {"B3 logit": lab[b3p.argmax(-1)]}
    for var in todo:
        guess[f"encoder {var}"] = lab[probs[var].argmax(-1)]
        for w in (0.5, 0.25, 0.75):  # 0.5 is the pre-set headline; the others are post hoc sensitivity
            tag = "" if w == 0.5 else " (post hoc)"
            guess[f"B3 + encoder {var}, w={w}{tag}"] = lab[((1 - w) * b3p + w * probs[var]).argmax(-1)]
    pairs = [(m, "B3 logit") for m in guess if m != "B3 logit"]
    if len(todo) == 2:
        pairs.append(("encoder marked", "encoder plain"))
    # Sentences annotated for several APCs appear as identical text with different labels. A text-only
    # model cannot separate them (B3's per-clause features can), so score the other rows on their own.
    amb = (df.groupby("sentence_norm")[features.target(df).name].transform("nunique") > 1).to_numpy()
    rows = []
    for tag, m in (("", np.ones(len(y), bool)), (" (text-unambiguous rows)", ~amb)):
        rows += [summarise(name + tag, p, y, g, labels, m) for name, p in guess.items()]
        draws = group_draws(g[m])
        for a, b in pairs:
            d, lo, hi = paired(y[m], guess[a][:, m], guess[b][:, m], labels, draws)
            rows.append({"model": f"{a} - {b} (paired){tag}", "n": int(m.sum()), "macro_f1": d, "ci_lo": lo, "ci_hi": hi})
    tab = ROOT / "results/tables"
    pd.DataFrame(rows).round(4).to_csv(tab / "t1_encoder.csv", index=False)
    pd.DataFrame(chosen).to_csv(tab / "t1_encoder_config.csv", index=False)

    pred_rows = pd.concat([pd.DataFrame({
        "variant": var, "id": np.tile(df.ID.to_numpy(), n_seeds), "seed": np.repeat(np.arange(n_seeds), len(y)),
        "true": np.tile(y, n_seeds), "pred": guess[f"encoder {var}"].ravel(),
        **{f"p_{c}": probs[var][:, :, j].ravel() for j, c in enumerate(labels)}}) for var in todo])
    pred_rows.round(5).to_csv(tab / "t1_encoder_preds.csv", index=False)
    if "plain" not in todo:
        return

    # ---- T3 (plain model): change in the predicted distribution under each perturbation ----
    base = probs["plain"].mean(0)
    true_j = np.array([labels.index(v) for v in y])
    p_true = lambda P: np.take_along_axis(P, true_j[None, :, None], 2)[..., 0]  # (seeds, n)
    rows, d_true = [], {}
    for v in VARIANTS:
        ok = ~np.isnan(vprobs[v][0, :, 0])
        delta = vprobs[v].mean(0)[ok] - base[ok]                       # (m, K)
        dt = p_true(vprobs[v]).mean(0) - p_true(probs["plain"]).mean(0)  # (n,), NaN where n/a
        d_true[v] = dt
        draws = group_draws(g[ok])
        row = {"perturbation": v, "n_rows": int(ok.sum())}
        for j, c in enumerate(labels):
            m, lo, hi = mean_ci(delta[:, j], draws)
            row.update({f"d_P_{c}": m, f"d_P_{c}_lo": lo, f"d_P_{c}_hi": hi})
        row["d_P_true"], row["d_P_true_lo"], row["d_P_true_hi"] = mean_ci(dt[ok], draws)
        rows.append(row)
    contrasts = [(f"mask_{x} - ctrl_{x}", f"mask_{x}", f"ctrl_{x}") for x in ("participle", "main_verb", "adverbials")]
    contrasts.append(("main_verb - participle", "mask_main_verb", "mask_participle"))
    for name, a, b in contrasts:
        both = ~np.isnan(d_true[a]) & ~np.isnan(d_true[b])
        m, lo, hi = mean_ci((d_true[a] - d_true[b])[both], group_draws(g[both]))
        rows.append({"perturbation": f"{name} (d_P_true)", "n_rows": int(both.sum()),
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

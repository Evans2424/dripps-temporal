"""T1 + T3: tuned xlm-roberta-base on the sentence, plain and with the two verbs marked, each also
with the annotated cues (B3's feature matrix) concatenated to the text vector (``*_feat``).

Needs a GPU and requirements-neural.txt (run on the lab machine, copy the tables back).
Hyperparameters are chosen inside each training fold (encoder.select_and_predict), folds are
B3's folds, and each (variant, seed, fold) is cached in results/cache_encoder/ so a crash in a
multi-hour run loses one fold. T3 perturbations are scored by the plain model's fold model, so
the ablation is out-of-fold and no checkpoint is kept. T1 also fuses B3 and encoder
probabilities (fixed average, nothing tuned) to ask whether text adds to the inventory; the
encoder is trained with class-weighted loss and B3 is not, so its posteriors are multiplied by the
training-fold class prior (and renormalised) before they are averaged.

Writes results/tables/t1_encoder.csv, t1_encoder_preds.csv, t1_encoder_config.csv,
t3_ablation.csv and t3_span_sample.csv (50 rows to check span recovery by hand).

  --variant V                   {plain|marked}[_feat|_res|_txt|_aux], or a group: both (plain + marked,
                                the default), all (those two plus their _feat twins), new (the _res, _txt
                                and _aux versions of both). Tables use every variant that is fully
                                cached, so run each variant, then any run rewrites the tables
  --checkpoint C                encoder.CHECKPOINTS key (xlmr, bertimbau, albertina, mdeberta, xlmr-large,
                                ...); a variant on a non-default checkpoint is named like marked@bertimbau
  --smoke                       100 rows, one seed, one config, one epoch, writes nothing
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_val_predict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dripps import ablate, encoder, evaluate, features, io, models  # noqa: E402
from dripps import schema  # noqa: E402
from dripps.schema import SEED  # noqa: E402

BASES = ["plain", "marked"]
GROUPS = {"both": BASES, "all": BASES + [f"{b}_feat" for b in BASES],
          "new": [f"{b}_{m}" for m in ("res", "txt", "aux") for b in BASES]}
#: annotated cue columns the auxiliary heads predict (the model never sees them as input)
AUX_COLS = ("ATMC", "ATSC", "TMC", "Position", "CNT")
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
        # an adverbial inside the APC verb group ("tendo já concluído") is part of the construction, not an
        # independent temporal cue, so it is neither masked nor drawn as a control
        adv = [a for a in ablate.adverbial_spans(t) if not (apc and apc[0][0] <= a[0] < apc[0][1])]
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
            out["ctrl_adverbials"][i] = ablate.mask_random(t, keep + adv, len(adv), mask, rng)
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


def _aspect_words(code):
    dyn, dur, tel = schema.ASPECT_PRIMITIVES[code]
    return f"{'dinâmico' if dyn else 'estativo'}, {'durativo' if dur else 'pontual'}, {'télico' if tel else 'atélico'}"


def _tense_words(code):
    t = schema.TAM_BUNDLE[code]
    flags = [w for k, w in (("perfective", "perfetivo"), ("progressive", "progressivo"), ("perfect", "perfeito"),
                            ("irrealis", "irrealis")) if t[k]]
    return " ".join([{"past": "passado", "present": "presente", "future": "futuro", "nonfinite": "não finito"}[t["tense"]],
                     *flags])


def cue_prefix(df):
    """The annotated cues as words (the Moens and Steedman primitives and the TAM bundle, not the corpus
    codes, which are subword soup to a pretrained model), put in front of the sentence for ``_txt``."""
    return [f"aspeto da oração principal: {_aspect_words(r.ATMC)}; tempo: {_tense_words(r.TMC)}; "
            f"aspeto da participial: {_aspect_words(r.ATSC)}; posição: {'final' if r.Position == 'Final' else 'não final'}; "
            f"conector: {'sim' if r.CNT.strip() else 'não'}. " for r in df.itertuples()]


def crossfit_offset(X, y, g, seed, prior):
    """B3 log-odds for every training row from a B3 fitted without that row's group, minus the log
    prior: the encoder is trained with class weights, so its offset has to be balanced too."""
    p = cross_val_predict(models.b3_logit(), X, y, cv=list(evaluate.folds(y, g, seed)), method="predict_proba")
    return np.log(p.clip(1e-6)) - np.log(prior)


def group_draws(groups, n_boot=2000):
    """Row indices of ``n_boot`` sentence-group bootstrap draws; built once per row subset."""
    rng = np.random.default_rng(SEED)
    return [evaluate.resample_group_rows(groups, rng) for _ in range(n_boot)]


def mean_ci(values, draws):
    """Mean of per-row ``values`` and its group-bootstrap 95% CI."""
    boot = [values[idx].mean() for idx in draws]
    return float(values.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def boot_f1(y, p, labels, draws, rs):
    """Macro-F1 of one model on each bootstrap draw (seed ``rs[d]`` for draw d), computed once per model."""
    return np.array([evaluate.macro_f1(y[idx], p[r][idx], labels) for idx, r in zip(draws, rs)])


def paired(y, a, b, labels, boots):
    """Observed macro-F1(a) - macro-F1(b) over (seeds, n) predictions, with a group-bootstrap CI;
    ``boots`` holds boot_f1 for a and b, so every model is resampled once, not once per contrast."""
    obs = evaluate.macro_f1_per_repeat(y, a, labels).mean() - evaluate.macro_f1_per_repeat(y, b, labels).mean()
    d = boots[0] - boots[1]
    return float(obs), float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


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
    ap.add_argument("--adopt", type=Path, help="old single-directory cache whose plain and marked folds to "
                    "copy into the per-variant caches (runs made before the per-variant cache keys)")
    ap.add_argument("--variant", default="both")
    ap.add_argument("--checkpoint", choices=list(encoder.CHECKPOINTS), default="xlmr")
    args = ap.parse_args()
    smoke = args.smoke
    ck = args.checkpoint
    if "@" in args.variant:
        sys.exit("name the checkpoint with --checkpoint, not in --variant")
    todo = [encoder.name(*encoder.parse(v)[:2], ck) for v in GROUPS.get(args.variant, [args.variant])]
    for v in todo:
        base, mode, _ = encoder.parse(v)
        if base not in BASES or mode not in ("", *encoder.MODES):
            sys.exit(f"unknown variant {v!r}")

    import torch
    if not smoke and not torch.cuda.is_available():
        sys.exit("no CUDA device: the tuned runs would take days on CPU. Run this on the GPU machine.")
    from transformers import AutoTokenizer
    mask = AutoTokenizer.from_pretrained(encoder.CHECKPOINTS['xlmr']).mask_token

    df = io.analysis_frame(io.load())
    if smoke:
        df = df.sample(100, random_state=SEED).reset_index(drop=True)
    texts = df.sentence_norm.tolist()
    y = features.target(df).to_numpy()
    g = features.groups(df).to_numpy()
    X = features.build(df).to_numpy()
    labels = sorted(pd.unique(y))
    n_seeds = 1 if smoke else N_SEEDS
    grid = lambda var: [dict(lr=3e-5, epochs=1)] if smoke else (  # noqa: E731
        encoder.CONFIGS_FEAT if encoder.parse(var)[1] == "feat" else encoder.CONFIGS)

    perturbed, span_df, marked, marks = build_variants(texts, mask)
    import transformers
    # a cached fold is only valid for the exact inputs, settings and library versions that produced it;
    # each variant is keyed on its own inputs, so adding or retuning one never discards another's folds
    prefix = cue_prefix(df)
    aux = np.stack([pd.factorize(df[c])[0] for c in AUX_COLS], 1)  # -1 where the cue is missing
    aux_sizes = [int(aux[:, j].max()) + 1 for j in range(aux.shape[1])]
    # text actually read by each variant: the sentence, or the sentence with the cue words in front
    # (marks move by the prefix length)
    cued = [p + t for p, t in zip(prefix, texts)]
    cued_marked = [p + t for p, t in zip(prefix, marked)]
    cued_marks = [tuple(None if m is None else m + len(p) for m in mk) for p, mk in zip(prefix, marks)]
    known = [encoder.name(b, m, c) for c in encoder.CHECKPOINTS for b in BASES for m in ("", *encoder.MODES)]
    extra = {"res": [np.asarray(X).round(6).tolist(), repr(models.b3_logit())],  # the offsets depend on B3's spec
             "aux": [aux.tolist(), list(AUX_COLS), encoder.AUX_WEIGHT]}
    cache = {}
    for v in known:
        b, m, ck_k = encoder.parse(v)
        ex = cued if m == "txt" and b == "plain" else cued_marked if m == "txt" else extra.get(m)
        tags = (["fp32 tf32"] if ck_k in encoder.FP32_CHECKPOINTS else []) + (  # numerics differ from folds cached before
            [["reviewed labels", y.tolist()]] if "review" in df.columns else [])  # labels and folds differ from the original export
        if tags:
            ex = [ex, *tags]
        cache[v] = ROOT / "results/cache_encoder" / encoder.cache_key(
            v, texts=texts, marked=marked, perturbed=perturbed, X=X, labels=labels, seed=SEED,
            library=transformers.__version__, extra=ex)
    if not smoke:
        for v in todo:
            cache[v].mkdir(parents=True, exist_ok=True)
    for var in ("plain", "marked") if args.adopt and not smoke else ():
        n = 0
        cache[var].mkdir(parents=True, exist_ok=True)
        for f in args.adopt.glob(f"{var}_s*_f*.npz"):
            dst = cache[var] / f.name.removeprefix(f"{var}_")
            if not dst.exists():
                shutil.copy(f, dst)
                n += 1
        print(f"adopted {n} {var} folds from {args.adopt}", flush=True)
    cached = lambda var: all((cache[var] / f"s{r}_f{k}.npz").exists()  # noqa: E731
                             for r in range(n_seeds) for k in range(evaluate.N_SPLITS))
    used = [v for v in known if v in todo or (not smoke and cached(v))]
    print(f"{len(df)} rows, {n_seeds} seeds, computing {todo} ({[len(grid(v)) for v in todo]} configs each), "
          f"tables from {used}", flush=True)
    print({v: sum(t is not None for t in perturbed[v]) for v in VARIANTS},
          "| marks:", sum(m[0] is not None for m in marks), "participle,",
          sum(m[1] is not None for m in marks), "main verb", flush=True)

    probs = {v: np.full((n_seeds, len(y), len(labels)), np.nan) for v in used}
    adj = {v: np.full_like(probs[v], np.nan) for v in used}  # encoder posteriors with the loss weighting undone
    b3p = np.full((n_seeds, len(y), len(labels)), np.nan)
    vprobs = {v: np.full_like(probs["plain"], np.nan) for v in VARIANTS} if "plain" in used else {}
    chosen = []
    for r in range(n_seeds):
        for k, (train, test) in enumerate(evaluate.folds(y, g, SEED + r)):
            b3 = models.b3_logit().fit(X[train], y[train])
            b3p[r, test] = b3.predict_proba(X[test])
            prior = np.array([(y[train] == c).mean() for c in labels])
            # B3's own scaling, fitted on the training fold (the inner validation rows inside
            # select_and_predict therefore help set the mean and std; unsupervised and small)
            Xtr, Xte = b3["scale"].transform(X[train]), b3["scale"].transform(X[test])
            off_tr = off_te = None  # B3 log-odds for the _res variants, built when first needed
            for var in used:
                f = cache[var] / f"s{r}_f{k}.npz"
                if f.exists() and not smoke:
                    z = np.load(f)
                else:
                    base, mode, ck_v = encoder.parse(var)
                    src = {("plain", "txt"): cued, ("marked", "txt"): cued_marked}.get(
                        (base, mode), texts if base == "plain" else marked)
                    mk = cued_marks if mode == "txt" else marks
                    if var == "plain":
                        where = {v: [i for i in test if perturbed[v][i] is not None] for v in VARIANTS}
                        ev = {"orig": [texts[i] for i in test],
                              **{v: [perturbed[v][i] for i in where[v]] for v in VARIANTS}}
                    else:
                        ev = {"orig": [src[i] for i in test]}
                    kw = {}
                    if base == "marked":
                        kw.update(train_marks=[mk[i] for i in train], eval_marks={"orig": [mk[i] for i in test]})
                    if mode == "feat":
                        kw.update(train_feats=Xtr, eval_feats={"orig": Xte})
                    if mode == "res":
                        if off_tr is None:
                            off_tr = crossfit_offset(X[train], y[train], g[train], SEED + r, prior)
                            off_te = np.log(b3p[r, test].clip(1e-6)) - np.log(prior)
                        kw.update(train_off=off_tr, eval_off={"orig": off_te})
                    if mode == "aux":
                        kw.update(train_aux=aux[train], aux_sizes=aux_sizes)
                    if ck_v != "xlmr":
                        kw.update(checkpoint=encoder.CHECKPOINTS[ck_v])
                    out, cfg = encoder.select_and_predict(
                        [src[i] for i in train], y[train], g[train], ev, labels=labels, seed=SEED + r,
                        configs=grid(var), **kw)
                    flat = {"orig": out["orig"], "cfg": np.array(json.dumps(cfg))}
                    if var == "plain":
                        for v in VARIANTS:
                            flat[f"i_{v}"], flat[f"p_{v}"] = np.array(where[v], dtype=int), out[v]
                    if not smoke:
                        np.savez(f.with_suffix(".tmp.npz"), **flat)
                        os.replace(f.with_suffix(".tmp.npz"), f)  # a crash mid-write must not leave a half file
                    z = flat
                probs[var][r, test] = z["orig"]
                adj[var][r, test] = z["orig"] * prior / (z["orig"] * prior).sum(1, keepdims=True)
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
    for var in used:
        guess[f"encoder {var}"] = lab[probs[var].argmax(-1)]
        if encoder.parse(var)[1] in ("feat", "res", "txt"):  # already see the cues; averaging with B3 would count them twice
            continue
        for w in (0.5, 0.25, 0.75):  # 0.5 is the pre-set headline; the others are post hoc sensitivity
            tag = "" if w == 0.5 else " (post hoc)"
            guess[f"B3 + encoder {var}, w={w}{tag}"] = lab[((1 - w) * b3p + w * adj[var]).argmax(-1)]
    pairs = [(m, "B3 logit") for m in guess if m != "B3 logit"]
    for v in used:  # a variant against its twin without the cues, its twin on the default checkpoint, and marked vs plain
        b, m, c = encoder.parse(v)
        for twin in (encoder.name(b, "", c) if m else None, encoder.name(b, m, "xlmr") if c != "xlmr" else None,
                     encoder.name("plain", m, c) if b == "marked" else None):
            if twin in used and (f"encoder {v}", f"encoder {twin}") not in pairs:
                pairs.append((f"encoder {v}", f"encoder {twin}"))
    # Sentences annotated for several APCs appear as identical text with different labels. A text-only
    # model cannot separate them (B3's per-clause features can), so score the other rows on their own.
    amb = (df.groupby("sentence_norm")[features.target(df).name].transform("nunique") > 1).to_numpy()
    rows = []
    for tag, m in (("", np.ones(len(y), bool)), (" (text-unambiguous rows)", ~amb)):
        rows += [summarise(name + tag, p, y, g, labels, m) for name, p in guess.items()]
        draws = group_draws(g[m])
        rng = np.random.default_rng(SEED)
        rs = [rng.integers(n_seeds) for _ in draws]  # the seed jitter folded in, as evaluate.bootstrap_group_ci does
        print(f"bootstrap: {len(guess)} models, {len(pairs)} paired contrasts{tag}", flush=True)
        boots = {k: boot_f1(y[m], guess[k][:, m], labels, draws, rs) for k in guess}
        for a, b in pairs:
            d, lo, hi = paired(y[m], guess[a][:, m], guess[b][:, m], labels, (boots[a], boots[b]))
            rows.append({"model": f"{a} - {b} (paired){tag}", "n": int(m.sum()), "macro_f1": d, "ci_lo": lo, "ci_hi": hi})
    tab = ROOT / "results/tables"
    pd.DataFrame(rows).round(4).to_csv(tab / "t1_encoder.csv", index=False)
    pd.DataFrame(chosen).to_csv(tab / "t1_encoder_config.csv", index=False)

    pred_rows = pd.concat([pd.DataFrame({
        "variant": var, "id": np.tile(df.ID.to_numpy(), n_seeds), "seed": np.repeat(np.arange(n_seeds), len(y)),
        "true": np.tile(y, n_seeds), "pred": guess[f"encoder {var}"].ravel(),
        **{f"p_{c}": probs[var][:, :, j].ravel() for j, c in enumerate(labels)}}) for var in used])
    pred_rows.round(5).to_csv(tab / "t1_encoder_preds.csv", index=False)
    if "plain" not in used:
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
    sample = found.sample(min(50, len(found)), random_state=SEED).assign(participle_ok="", main_verb_ok="")
    print(f"span recovery: participle {int((span_df.participle != '').sum())}, "
          f"main verb {int((span_df.main_verb != '').sum())}, both {len(found)} of {len(span_df)}")
    out = tab / "t3_span_sample.csv"
    if out.exists() and pd.read_csv(out).participle_ok.notna().any():
        print("t3_span_sample.csv already hand-labelled; left as is")
    else:
        sample.to_csv(out, index=False)
    print(pd.DataFrame(rows).round(3).to_string())


if __name__ == "__main__":
    main()

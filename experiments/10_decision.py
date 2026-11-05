"""T5: Jev-style decisions, scored on B3's folds and perturbed like the encoder's T3.

A stock LLM (or a diffusion LM behind an open-jev ``/v1/systemone`` server) is shown the sentence, the question
and three options in one prompt and returns one probability per option from a single forward pass (see
src/dripps/decision.py). Nothing is trained: raw rows use no labels; the calibrated rows fit a 12-parameter logit
on each training fold. The three option orders are averaged to cancel position bias.

  python experiments/10_decision.py --model qwen3.8-27b-fp8          # GPU; one model per call, tables use every cached one
  python experiments/10_decision.py --engine http --url http://host:8000/v1/systemone --name diffusiongemma --id google/diffusiongemma-26B-A4B-it
  python experiments/10_decision.py --smoke --model qwen3.5-0.8b     # 100 rows, one seed, temp cache, prints, writes nothing

Scores each of the 9 texts (plain, marked, and the 7 perturbations of 08_encoder.py) once per model into
results/cache_decision/<name>/<text>.npy (a crash loses one text), then writes results/tables/t5_decision.csv,
t5_decision_preds.csv, t5_decision_perturb.csv and t5_decision_diag.csv. Headline: the first model of MODELS, plain
text, template 1, raw (when it is not cached on this machine nothing is called the headline). Everything else is
exploratory, no multiplicity correction. A model run with --template 2 needs its own --name (e.g. qwen3.8-27b-t2).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from dripps import decision, evaluate, features, io, models  # noqa: E402
from dripps.schema import SEED  # noqa: E402

spec = importlib.util.spec_from_file_location("enc08", ROOT / "experiments/08_encoder.py")
e08 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(e08)  # 08 imports torch and spaCy lazily, so this is light

#: name used in the tables -> Hugging Face id. The first is the headline. FP8 because 27B in bf16 (about 54 GB) does not fit a 40 GB GPU.
MODELS = {"qwen3.8-27b-fp8": "Qwen/Qwen3.8-27B-FP8", "qwen3.6-27b-fp8": "Qwen/Qwen3.6-27B-FP8",
          "qwen3.5-9b": "Qwen/Qwen3.5-9B", "qwen3.5-4b": "Qwen/Qwen3.5-4B", "qwen3.5-0.8b": "Qwen/Qwen3.5-0.8B"}
#: RQ2 grid: Portuguese LLMs that differ mainly in training variety, plus multilingual controls. All run with the
#: Portuguese template 2 (docs/VARIETY_MODELS_PLAN.md); confirmatory pair: the two Gervásio-7B models.
VARIETY_MODELS = {
    "gervasio-7b-ptpt-t2": "PORTULAN/gervasio-7b-portuguese-ptpt-decoder",
    "gervasio-7b-ptbr-t2": "PORTULAN/gervasio-7b-portuguese-ptbr-decoder",
    "gervasio-8b-ptpt-t2": "PORTULAN/gervasio-8b-portuguese-ptpt-decoder",
    "amalia-9b-t2": "amalia-llm/AMALIA-9B-0626-DPO",
    "bode-7b-t2": "recogna-nlp/bode-7b-alpaca-pt-br-no-peft",
    "sabia-7b-t2": "maritaca-ai/sabia-7b",
    "tucano-2b4-t2": "TucanoBR/Tucano-2b4-Instruct",
    "eurollm-9b-t2": "utter-project/EuroLLM-9B-Instruct",
    "salamandra-7b-t2": "BSC-LT/salamandra-7b-instruct",
    "qwen3.5-9b-t2": "Qwen/Qwen3.5-9B",
}
MASK = "<mask>"  # what the encoder run's perturbations used (the XLM-R mask token)
TEXTS = ["plain", "marked", *e08.VARIANTS]
N_SEEDS = e08.N_SEEDS


def server_version(url: str, tries: int = 5, pause: float = 10.0) -> dict | None:
    """GET <origin>/v1/version of an open decision server (OpenJev's shim answers it). None only when the server
    answers that it has no such endpoint (404 or a non-object); a slow or unreachable server raises after ``tries``,
    so a hiccup is never recorded as "no version" (which would make the next resume refuse the cache)."""
    import http.client
    import time
    import urllib.error
    import urllib.parse
    import urllib.request
    u = urllib.parse.urlsplit(url)
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(f"{u.scheme}://{u.netloc}/v1/version", timeout=30) as r:
                v = json.loads(r.read())
            return {k: v.get(k) for k in ("model_dir", "T", "noul_t", "noul_bias", "flags", "shim_sha256")} \
                if isinstance(v, dict) else None
        except urllib.error.HTTPError as e:
            if e.code in (404, 405, 501):
                return None
            err = e
        except (OSError, ValueError, http.client.HTTPException) as e:
            err = e
        if attempt < tries - 1:
            time.sleep(pause)
    raise RuntimeError(f"{url}: /v1/version did not answer after {tries} tries: {err!r}")


def checkpoint_id(path: Path) -> str:
    """A local checkpoint's identity: every small file in full, plus the size and the first and last MiB of each
    large one (the weights). Names and sizes alone are the same for any model of the same shape."""
    h = hashlib.sha1()
    for f in sorted(p for p in path.rglob("*") if p.is_file() and ".cache" not in p.parts):
        n = f.stat().st_size
        h.update(f"{f.relative_to(path)}:{n}".encode())
        with f.open("rb") as fh:
            if n <= 8 << 20:
                h.update(fh.read())
            else:
                h.update(fh.read(1 << 20))
                fh.seek(-(1 << 20), 2)
                h.update(fh.read())
    return h.hexdigest()[:10]


def load_dotenv(path: Path = ROOT / ".env") -> None:
    """KEY=VALUE lines into os.environ; a variable already set wins. ``export`` is allowed, a value in matching
    quotes is literal, an unquoted value ends at `` #`` (an inline comment), and comment lines start with ``#``."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.removeprefix("export ").split("=", 1)
            v = v.strip()
            if len(v) >= 2 and v[0] in "'\"" and v[-1] == v[0]:
                v = v[1:-1]  # a quoted value is taken literally, '#' included
            elif v[:1] not in "'\"":
                v = v.split(" #", 1)[0].split("\t#", 1)[0].strip()  # an unquoted value ends at an inline comment
            os.environ.setdefault(k.strip(), v)  # an unmatched quote is kept as is, not half-stripped


def prob_metrics(P, Y):
    """log loss, Brier (sum of squares over the classes) and ECE (10 bins on the top probability), pooled over seeds."""
    p_true = np.take_along_axis(P, np.broadcast_to(Y, P.shape[:-1])[..., None], -1)[..., 0]
    onehot = np.eye(P.shape[-1])[np.broadcast_to(Y, P.shape[:-1])]
    conf, hit = P.max(-1).ravel(), (P.argmax(-1) == np.broadcast_to(Y, P.shape[:-1])).ravel()
    bins = np.minimum((conf * 10).astype(int), 9)
    ece = sum(abs(hit[bins == b].mean() - conf[bins == b].mean()) * (bins == b).mean() for b in range(10) if (bins == b).any())
    return {"log_loss": -np.log(np.clip(p_true, 1e-12, 1)).mean(), "brier": ((P - onehot) ** 2).sum(-1).mean(), "ece10": ece}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help=f"one of {list(MODELS) + list(VARIETY_MODELS)}; omit to only rebuild the tables")
    ap.add_argument("--name", help="table name for a model not in MODELS (use with --id)")
    ap.add_argument("--id", help="Hugging Face id (or the server's model name for --engine http)")
    ap.add_argument("--engine", choices=("hf", "http", "fake"), default="hf", help="fake: random probabilities, to check the pipeline only")
    ap.add_argument("--url")
    ap.add_argument("--template", type=int, default=1, choices=sorted(decision.TEMPLATES))
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    smoke = args.smoke or args.engine == "fake"  # random probabilities: pipeline check only, writes nothing
    if args.engine == "http" and not args.url:
        sys.exit("--engine http needs --url")
    if args.engine == "http":
        load_dotenv()  # TYPESAFE_API_KEY for the Jev API
    name = args.name or args.model
    if args.model in VARIETY_MODELS and args.template != 2:
        sys.exit(f"{args.model} is a template-2 grid model: add --template 2")
    cache_root = Path(tempfile.mkdtemp()) if smoke else ROOT / "results/cache_decision"

    df = io.analysis_frame(io.load())
    if smoke:
        df = df.sample(100, random_state=SEED).reset_index(drop=True)
    texts, y, g, X = df.sentence_norm.tolist(), features.target(df).to_numpy(), features.groups(df).to_numpy(), features.build(df).to_numpy()
    labels = list(decision.LABELS)
    assert sorted(pd.unique(y)) == labels
    n_seeds = 1 if smoke else N_SEEDS
    perturbed, _, marked, _ = e08.build_variants(texts, MASK)
    src = {"plain": texts, "marked": marked, **perturbed}
    sha = hashlib.sha1(json.dumps([src, MASK, labels, SEED], ensure_ascii=False, sort_keys=True).encode()).hexdigest()[:10]
    code = "".join(inspect.getsource(f) for f in (decision.build_prompt, decision.render_user, decision.render_chat, decision.render_plain, decision.per_order, decision.letter_ids))
    # what the HTTP engine sends is defined by http_body and the option names, not by the chat rendering
    code_http = "".join(inspect.getsource(f) for f in (decision.build_prompt, decision.per_order, decision.http_body)) \
        + json.dumps(decision.OPTION_NAMES)

    def prompt_sha(t, engine="hf"):
        c = code_http if engine == "http" else code  # the hf formula is unchanged, so earlier hf caches stay valid
        return hashlib.sha1(json.dumps([decision.TEMPLATES[t], decision.CYCLES, decision.LETTERS, t, c]).encode()).hexdigest()[:10]

    if name:
        mid = args.id or {**MODELS, **VARIETY_MODELS}.get(args.model, name)
        d = cache_root / name
        d.mkdir(parents=True, exist_ok=True)
        meta = {"id": mid, "engine": args.engine, "template": args.template, "texts": sha,
                "prompt": prompt_sha(args.template, args.engine)}
        old = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else None
        if args.engine == "hf" and Path(mid).is_dir():  # a local checkpoint (e.g. OpenJev): pin its files, not a commit
            meta["revision"] = None
            meta["files"] = checkpoint_id(Path(mid))
        elif args.engine == "hf":
            if old:  # the revision is pinned by the first run, so a resumed run never needs the network or a newer commit
                meta["revision"] = old.get("revision")
            else:
                from huggingface_hub import constants
                if constants.HF_HUB_OFFLINE:  # HPC compute node: pin the commit the local cache holds (raises if none)
                    from huggingface_hub import snapshot_download
                    meta["revision"] = Path(snapshot_download(mid, local_files_only=True)).name
                else:  # online: any failure is real (bad token, Hub down), never a silent fallback
                    try:
                        from huggingface_hub import HfApi
                        meta["revision"] = HfApi().model_info(mid).sha
                    except Exception:
                        if not smoke:  # a run without a pinned revision would load whatever main points to later
                            raise
                        meta["revision"] = None  # a smoke run (writes nothing) may go on without the Hub
        if args.engine == "hf":
            import transformers
            meta["transformers"] = transformers.__version__
        open_server = args.engine == "http" and not decision.is_typesafe(args.url)
        if open_server:
            meta["server"] = server_version(args.url)  # e.g. OpenJev's shim: weights dir, readout temperature, shim hash
            meta["fingerprint"] = (old or {}).get("fingerprint")  # the answers' model string, pinned by the first run
        if old and old != meta:
            sys.exit(f"{d} was made with other settings ({(d / 'meta.json').read_text()}); delete it or pick another --name")
        (d / "meta.json").write_text(json.dumps(meta))
        todo = [v for v in TEXTS if not (d / f"{v}.npy").exists()]
        score = None
        seen = {"model": meta["fingerprint"]} if open_server and meta.get("fingerprint") else {}  # resume: same server
        for v in todo:
            if score is None:
                score = {"hf": lambda: decision.hf_engine(mid, batch=args.batch, revision=meta["revision"]),
                         "http": lambda: (lambda ps: decision.score_http(args.url, ps, model=args.id, seen=seen)),
                         "fake": lambda: (lambda ps: np.random.default_rng(len(ps)).dirichlet([1, 1, 1], len(ps)))}[args.engine]()
            ix = [i for i, t in enumerate(src[v]) if t is not None]
            P = np.full((len(decision.CYCLES), len(texts), len(labels)), np.nan)
            P[:, ix] = decision.per_order(score, [src[v][i] for i in ix], template=args.template, marked=v == "marked")
            np.save(d / f"{v}.tmp.npy", P)
            os.replace(d / f"{v}.tmp.npy", d / f"{v}.npy")  # a kill mid-write must not leave a half file
            if open_server and seen.get("model") and meta.get("fingerprint") != seen["model"]:
                meta["fingerprint"] = seen["model"]  # written with the first scored text, checked on every resume
                (d / "meta.json").write_text(json.dumps(meta))
            if np.isnan(P[:, ix]).any():
                sys.exit(f"{name} {v}: NaN probabilities")
            agree = np.mean([len({int(P[k, i].argmax()) for k in range(len(decision.CYCLES))}) == 1 for i in ix])
            print(f"scored {name} {v}: {len(ix)} texts ({score.format if hasattr(score, 'format') else args.engine}"
                  f" prompt), the 3 option orders agree on {agree:.0%}", flush=True)
        if args.engine == "hf" and not (d / "nll.npy").exists():  # RQ2 manipulation check: familiarity by variety
            if score is None:
                score = decision.hf_engine(mid, batch=args.batch, revision=meta["revision"])
            np.save(d / "nll.npy", score.nll(texts))
            print(f"saved {name} sentence NLL", flush=True)

    # ---- tables from every complete cache whose texts match ----
    names = [p.parent.name for p in sorted(cache_root.glob("*/meta.json"))
             if json.loads(p.read_text())["texts"] == sha and json.loads(p.read_text())["prompt"] == prompt_sha(json.loads(p.read_text())["template"], json.loads(p.read_text())["engine"]) and all((p.parent / f"{v}.npy").exists() for v in TEXTS)]
    names = sorted(names, key=lambda n: list(MODELS).index(n) if n in MODELS else len(MODELS))
    if not names:
        print(f"no complete cache matches the current texts and prompts in {cache_root}; tables not written")
        return
    raw = {n: {v: np.load(cache_root / n / f"{v}.npy") for v in TEXTS} for n in names}  # (orders, n, 3), NaN where n/a
    Y = np.array([labels.index(v) for v in y])
    pred = lambda P: np.array(labels)[P.argmax(-1)]  # noqa: E731
    rep = lambda P: np.broadcast_to(P, (n_seeds, *P.shape)).copy()  # noqa: E731  label-free, so identical across seeds

    b3p, probs = np.full((n_seeds, len(y), 3), np.nan), {}
    for n in names:
        probs[f"decision {n}"] = rep(decision.debias(raw[n]["plain"]))
        probs[f"decision {n} marked"] = rep(decision.debias(raw[n]["marked"]))
        for t in ("", " marked"):
            probs[f"decision {n}{t} calibrated"] = np.full((n_seeds, len(y), 3), np.nan)
    for r in range(n_seeds):
        for tr, te in evaluate.folds(y, g, SEED + r):
            b3p[r, te] = models.b3_logit().fit(X[tr], y[tr]).predict_proba(X[te])
            for n in names:
                for t in ("", " marked"):
                    P = probs[f"decision {n}{t}"][r]
                    probs[f"decision {n}{t} calibrated"][r, te] = decision.calibrate(P[tr], y[tr], P[te])
    for n in names:
        for t in ("", " marked"):
            probs[f"B3 + decision {n}{t} calibrated, w=0.5"] = (b3p + probs[f"decision {n}{t} calibrated"]) / 2
    allp = {"B3 logit": b3p, **probs}
    guess = {k: pred(P) for k, P in allp.items()}

    head = next(iter(MODELS)) if next(iter(MODELS)) in names else None  # model-vs-headline pairs need the real headline
    pairs = [(k, "B3 logit") for k in probs]
    for n in names:
        for c in ("", " calibrated"):
            pairs.append((f"decision {n} marked{c}", f"decision {n}{c}"))
            if head and n != head:
                pairs.append((f"decision {n}{c}", f"decision {head}{c}"))
    pairs = [(a, b) for a, b in pairs if a in guess and b in guess and a != b]
    amb = (df.groupby("sentence_norm")[features.target(df).name].transform("nunique") > 1).to_numpy()
    rows = []
    for tag, m in (("", np.ones(len(y), bool)), (" (text-unambiguous rows)", ~amb)):
        for k, p in guess.items():
            row = e08.summarise(k + tag, p, y, g, labels, m)
            row.update(prob_metrics(allp[k][:, m], Y[m]))  # B3 too, so the probability quality can be compared
            rows.append(row)
        draws = e08.group_draws(g[m])
        rng = np.random.default_rng(SEED)
        rs = [rng.integers(n_seeds) for _ in draws]
        boots = {k: e08.boot_f1(y[m], guess[k][:, m], labels, draws, rs) for k in guess}
        for a, b in pairs:
            d, lo, hi = e08.paired(y[m], guess[a][:, m], guess[b][:, m], labels, (boots[a], boots[b]))
            rows.append({"model": f"{a} - {b} (paired){tag}", "n": int(m.sum()), "macro_f1": d, "ci_lo": lo, "ci_hi": hi})
    t1 = pd.DataFrame(rows).round(4)

    # ---- T3-style perturbations, raw probabilities of each model ----
    pert, diag = [], []
    gold_c = models.b3_logit().fit(X, y)["clf"].coef_
    true_j = Y
    for n in names:
        base = decision.debias(raw[n]["plain"])
        p_true = lambda P: P[np.arange(len(y)), true_j]  # noqa: E731
        d_true = {}
        for v in e08.VARIANTS:
            Pv = decision.debias(raw[n][v])
            ok = ~np.isnan(Pv[:, 0])
            delta = Pv[ok] - base[ok]
            d_true[v] = p_true(Pv) - p_true(base)
            draws = e08.group_draws(g[ok])
            row = {"model": n, "perturbation": v, "n_rows": int(ok.sum())}
            for j, c in enumerate(labels):
                row.update(dict(zip((f"d_P_{c}", f"d_P_{c}_lo", f"d_P_{c}_hi"), e08.mean_ci(delta[:, j], draws))))
            row["d_P_true"], row["d_P_true_lo"], row["d_P_true_hi"] = e08.mean_ci(d_true[v][ok], draws)
            pert.append(row)
        for nm, a, b in [(f"mask_{x} - ctrl_{x}", f"mask_{x}", f"ctrl_{x}") for x in ("participle", "main_verb", "adverbials")] + \
                [("main_verb - participle", "mask_main_verb", "mask_participle")]:
            both = ~np.isnan(d_true[a]) & ~np.isnan(d_true[b])
            mn, lo, hi = e08.mean_ci((d_true[a] - d_true[b])[both], e08.group_draws(g[both]))
            pert.append({"model": n, "perturbation": f"{nm} (d_P_true)", "n_rows": int(both.sum()), "d_P_true": mn, "d_P_true_lo": lo, "d_P_true_hi": hi})

        # diagnostics: position bias before debiasing, and whether the decision weights the cues like the corpus does
        Po = raw[n]["plain"]
        for k, order in enumerate(decision.CYCLES):
            top = Po[k].argmax(-1)
            diag.append({"model": n, "metric": f"order{k} accuracy", "value": float((top == Y).mean())})
            for s in range(3):
                diag.append({"model": n, "metric": f"order{k} share of slot {decision.LETTERS[s]}",
                             "value": float(np.mean([order.index(t) == s for t in top]))})
        tops = Po.argmax(-1)
        diag.append({"model": n, "metric": "all three orders agree", "value": float((tops == tops[0]).all(0).mean())})
        llm = pred(base)
        if len(set(llm)) == 3:
            llm_c = models.b3_logit().fit(X, llm)["clf"].coef_
            diag.append({"model": n, "metric": "spearman of cue coefficients, LLM labels vs gold labels",
                         "value": float(spearmanr(gold_c.ravel(), llm_c.ravel())[0])})
            at = 0
            for blk, cols in features.CUE_BLOCKS.items():
                gs, ls = np.abs(gold_c[:, at:at + len(cols)]).sum(), np.abs(llm_c[:, at:at + len(cols)]).sum()
                diag += [{"model": n, "metric": f"share of |coef| in {blk}, gold", "value": float(gs / np.abs(gold_c).sum())},
                         {"model": n, "metric": f"share of |coef| in {blk}, LLM", "value": float(ls / np.abs(llm_c).sum())}]
                at += len(cols)
        diag.append({"model": n, "metric": "agreement with B3 (out-of-fold, seed 0)", "value": float((llm == guess["B3 logit"][0]).mean())})

    pr = pd.concat([pd.DataFrame({"variant": k, "id": np.tile(df.ID.to_numpy(), n_seeds), "seed": np.repeat(np.arange(n_seeds), len(y)),
                                  "true": np.tile(y, n_seeds), "pred": guess[k].ravel(),
                                  **{f"p_{c}": probs[k][:, :, j].ravel() for j, c in enumerate(labels)}}) for k in probs])
    if smoke:
        pd.set_option("display.width", 200, "display.max_colwidth", 70)
        print(t1[~t1.model.str.contains("unambiguous")].head(12).to_string(), pd.DataFrame(diag).to_string(), sep="\n")
        return
    tab = ROOT / "results/tables"
    t1.to_csv(tab / "t5_decision.csv", index=False)
    pr.round(5).to_csv(tab / "t5_decision_preds.csv", index=False)
    pd.DataFrame(pert).round(4).to_csv(tab / "t5_decision_perturb.csv", index=False)
    pd.DataFrame(diag).round(4).to_csv(tab / "t5_decision_diag.csv", index=False)
    print(f"tables from {names}", flush=True)


if __name__ == "__main__":
    main()

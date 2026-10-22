"""Fine-tune a multilingual encoder on a sentence, optionally pooling at marked verbs (T1).

Plain torch. ``fit_predict`` takes fixed hyperparameters; ``select_and_predict``
chooses them with one sentence-grouped split *inside* the training fold, so
nothing is ever tuned on the held-out fold. ``torch`` and ``transformers`` are
imported inside ``fit_predict`` so the rest of the package (and ``make test``)
works without them.
"""

from __future__ import annotations

import numpy as np

from . import evaluate

#: short name -> Hugging Face checkpoint. The default (xlmr) is what every earlier table used.
CHECKPOINTS = {
    "xlmr": "xlm-roberta-base",
    "xlmr-large": "FacebookAI/xlm-roberta-large",
    "mdeberta": "microsoft/mdeberta-v3-base",
    "bertimbau": "neuralmind/bert-base-portuguese-cased",
    "bertimbau-large": "neuralmind/bert-large-portuguese-cased",
    "albertina": "PORTULAN/albertina-100m-portuguese-ptpt-encoder",
    "albertina-900m": "PORTULAN/albertina-900m-portuguese-ptpt-encoder",
}
CHECKPOINT = CHECKPOINTS["xlmr"]
#: How the annotated cues reach the model (a variant is ``{plain|marked}[_mode][@checkpoint]``):
#:   feat  cue matrix concatenated to the text vector (a second linear layer, own learning rate)
#:   res   the text head learns a correction to B3's cross-fitted log-odds (product of experts)
#:   txt   the cues written as words in front of the sentence
#:   aux   the text vector must also predict the cues (auxiliary loss); no cues at test time
MODES = ("feat", "res", "txt", "aux")
AUX_WEIGHT = 0.5
CONFIGS = [dict(lr=lr, epochs=ep) for lr in (3e-5, 5e-5) for ep in (6, 12)]
# feat_lr: the annotated-feature weights start from zero and have only ~600 steps to move, so they get
# their own, larger rate (the encoder keeps lr). The default is only a fallback: the feature variants
# tune it (CONFIGS_FEAT), because too low ignores the cues and too high lets them drown the text.
HPARAMS = dict(batch_size=16, max_len=256, warmup=0.1, feat_lr=1e-3)
# the text-only grid crossed with the feature rate, so a feature variant and its text-only twin differ
# in the cues and not in tuning budget
CONFIGS_FEAT = [dict(**c, feat_lr=fl) for c in CONFIGS for fl in (1e-4, 1e-3, 1e-2)]


def parse(var):
    """'marked_feat@bertimbau' -> ('marked', 'feat', 'bertimbau'); mode is '' for text only."""
    head, _, ck = var.partition("@")
    base, _, mode = head.partition("_")
    return base, mode, ck or "xlmr"


def name(base, mode="", ck="xlmr"):
    return base + (f"_{mode}" if mode else "") + (f"@{ck}" if ck != "xlmr" else "")


def cache_key(var, *, texts, marked, perturbed, X, labels, seed, library, extra=None):
    """Cache directory name for one variant. Each variant is keyed on its own inputs only, so changing
    the feature grid or the cue matrix never orphans the cached plain and marked folds. ``extra``
    holds what only the newer modes read (serialised cues, auxiliary targets); None leaves the key
    of the earlier variants exactly as it was."""
    import hashlib
    import json
    base, mode, ck = parse(var)
    feat = mode == "feat"
    hp = {k: v for k, v in HPARAMS.items() if feat or k != "feat_lr"}
    parts = [var, labels, CONFIGS_FEAT if feat else CONFIGS, hp, CHECKPOINTS[ck], seed, library]
    parts += [texts, perturbed] if var == "plain" else [marked] if base == "marked" else [texts]
    if feat:
        parts.append(np.asarray(X).round(6).tolist())
    if extra is not None:
        parts.append(extra)
    return f"{var}_" + hashlib.sha1(json.dumps(parts).encode()).hexdigest()[:10]


def fit_predict(train_texts, y_train, eval_texts: dict, *, labels, seed, lr, epochs,
                train_marks=None, eval_marks: dict | None = None, train_feats=None,
                eval_feats: dict | None = None, train_off=None, eval_off: dict | None = None,
                train_aux=None, aux_sizes=(), checkpoint=CHECKPOINT, **hp):
    """Fit on ``train_texts``; return class probabilities for each named list of texts.

    ``*_marks`` hold, per text, the character offsets of the tokens to pool at (None = missing,
    which falls back to the first token). The head sees [first token, pooled marks...]. Columns
    follow ``labels``; cross-entropy is class-weighted because the metric is macro-F1.

    ``*_feats`` are (n, d) arrays of annotated features, already scaled. They join the
    head input next to the text vector. That is one linear layer over the concatenation, written as
    two layers whose logits add, so the feature block can have its own learning rate.

    ``*_off`` are (n, K) log-odds added to the head's logits (B3's, cross-fitted, so the text head
    learns a correction to them; the head then starts at zero, i.e. at B3). ``train_aux`` is an
    (n, m) integer array of cue classes (-1 = missing) predicted from the text vector by one small
    head each, with weight AUX_WEIGHT; they only shape the encoder and are not used to predict.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

    hp = {**HPARAMS, **hp}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tok = AutoTokenizer.from_pretrained(checkpoint)
    assert tok.is_fast, "pooling at marks needs offset mappings, which only fast tokenizers give"
    enc = AutoModel.from_pretrained(checkpoint, dtype=torch.float32).to(dev)
    n_marks = len(train_marks[0]) if train_marks else 0
    assert not n_marks or eval_marks is not None, "trained with marks but no eval marks"
    width = enc.config.hidden_size * (1 + n_marks)
    head = torch.nn.Sequential(torch.nn.Dropout(0.1), torch.nn.Linear(width, len(labels))).to(dev)
    assert train_off is None or eval_off is not None, "trained with offsets but no eval offsets"
    if train_off is not None:
        torch.nn.init.zeros_(head[1].weight); torch.nn.init.zeros_(head[1].bias)  # start at the offset
    aux_heads = torch.nn.ModuleList([torch.nn.Linear(width, c) for c in aux_sizes]).to(dev)
    n_feat = 0 if train_feats is None else train_feats.shape[1]
    assert not n_feat or eval_feats is not None, "trained with features but no eval features"
    assert not n_feat or set(eval_feats) == set(eval_texts), "eval_feats needs one array per eval text set"
    feat_head = None
    if n_feat:
        feat_head = torch.nn.Linear(n_feat, len(labels), bias=False).to(dev)
        torch.nn.init.zeros_(feat_head.weight)
    y = torch.tensor([labels.index(v) for v in y_train])
    counts = torch.bincount(y, minlength=len(labels)).float()
    loss_fn = torch.nn.CrossEntropyLoss(weight=(len(y) / (len(labels) * counts.clamp(min=1))).to(dev))
    n, bs = len(y), hp["batch_size"]
    steps = epochs * ((n + bs - 1) // bs)
    params = list(enc.parameters()) + list(head.parameters()) + list(aux_heads.parameters())
    groups = [dict(params=params, lr=lr)]
    if n_feat:
        groups.append(dict(params=list(feat_head.parameters()), lr=hp["feat_lr"]))
        params = params + list(feat_head.parameters())  # clipped with the rest
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(hp["warmup"] * steps), steps)

    def logits(texts, marks, feats, offset=None):
        b = tok(list(texts), padding=True, truncation=True, max_length=hp["max_len"],
                return_offsets_mapping=True, return_tensors="pt")
        spans = b.pop("offset_mapping")
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            h = enc(**{k: v.to(dev) for k, v in b.items()}).last_hidden_state
        parts = [h[:, 0]]
        for j in range(n_marks):
            idx = []
            for i in range(len(texts)):
                c = marks[i][j] if marks is not None else None
                hit = [] if c is None else ((spans[i, :, 0] <= c) & (spans[i, :, 1] > c)).nonzero().flatten().tolist()
                idx.append(hit[0] if hit else 0)
            parts.append(h[torch.arange(len(texts)), torch.tensor(idx, device=dev)])
        z = torch.cat(parts, -1).float()
        out = head(z)
        if n_feat:
            out = out + feat_head(torch.tensor(feats, dtype=torch.float32, device=dev))
        if offset is not None:
            out = out + torch.as_tensor(offset, dtype=torch.float32, device=dev)
        return out, z

    aux_y = None if train_aux is None else torch.tensor(np.asarray(train_aux), dtype=torch.long)
    for _ in range(epochs):
        enc.train(); head.train(); aux_heads.train()
        order = rng.permutation(n)
        for i in range(0, n, bs):
            ix = order[i:i + bs]
            out, z = logits([train_texts[j] for j in ix], [train_marks[j] for j in ix] if train_marks else None,
                            train_feats[ix] if n_feat else None, train_off[ix] if train_off is not None else None)
            loss = loss_fn(out, y[ix].to(dev))
            for j, h in enumerate(aux_heads):  # cue j from the text vector; -1 rows are unannotated
                t = aux_y[ix, j].to(dev)
                if (t >= 0).any():
                    loss = loss + AUX_WEIGHT / len(aux_heads) * torch.nn.functional.cross_entropy(
                        h(z), t, ignore_index=-1)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad()

    enc.eval(); head.eval()
    probs = {}
    with torch.no_grad():
        for name, texts in eval_texts.items():
            m = eval_marks[name] if eval_marks else None
            f = eval_feats[name] if n_feat else None
            o = eval_off[name] if train_off is not None else None
            chunks = [logits(texts[i:i + 64], m[i:i + 64] if m else None, f[i:i + 64] if n_feat else None,
                             o[i:i + 64] if o is not None else None)[0].softmax(-1).cpu().numpy()
                      for i in range(0, len(texts), 64)]
            probs[name] = np.concatenate(chunks) if chunks else np.empty((0, len(labels)))
    return probs


def select_and_predict(train_texts, y_train, groups_train, eval_texts, *, labels, seed,
                       configs=CONFIGS, train_marks=None, eval_marks=None, train_feats=None, eval_feats=None,
                       train_off=None, eval_off=None, train_aux=None, aux_sizes=(), fit=fit_predict, **fit_kw):
    """Pick the config with the best macro-F1 on one grouped inner split of the training rows,
    refit it on all of them, and predict ``eval_texts``. Returns (probs, chosen config)."""
    y_train = np.asarray(y_train)
    inner_fit, inner_val = next(iter(evaluate.folds(y_train, groups_train, seed)))
    take = lambda xs, ix: None if xs is None else [xs[i] for i in ix]  # noqa: E731
    take_feats = lambda xs, ix: None if xs is None else xs[ix]  # noqa: E731
    # offsets and auxiliary targets are passed only when present, so a plain fit keeps the old signature
    def extra(ix, val=None):
        kw = {}
        if train_off is not None:
            kw.update(train_off=take_feats(train_off, ix), eval_off=None if val is None else {"val": take_feats(train_off, val)})
        if train_aux is not None:
            kw.update(train_aux=take_feats(train_aux, ix), aux_sizes=aux_sizes)
        return kw

    scores = []
    for cfg in configs:
        p = fit(take(train_texts, inner_fit), y_train[inner_fit], {"val": take(train_texts, inner_val)},
                labels=labels, seed=seed, train_marks=take(train_marks, inner_fit),
                eval_marks=None if train_marks is None else {"val": take(train_marks, inner_val)},
                train_feats=take_feats(train_feats, inner_fit),
                eval_feats=None if train_feats is None else {"val": take_feats(train_feats, inner_val)},
                **extra(inner_fit, inner_val), **fit_kw, **cfg)["val"]
        scores.append(evaluate.macro_f1(y_train[inner_val], np.array(labels)[p.argmax(-1)], labels))
    best = configs[int(np.argmax(scores))]
    final = {}
    if train_off is not None:
        final.update(train_off=train_off, eval_off=eval_off)
    if train_aux is not None:
        final.update(train_aux=train_aux, aux_sizes=aux_sizes)
    probs = fit(train_texts, y_train, eval_texts, labels=labels, seed=seed, train_marks=train_marks,
                eval_marks=eval_marks, train_feats=train_feats, eval_feats=eval_feats, **final, **fit_kw, **best)
    return probs, {**best, "inner_val_f1": float(max(scores))}

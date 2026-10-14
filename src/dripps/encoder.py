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

CHECKPOINT = "xlm-roberta-base"
CONFIGS = [dict(lr=lr, epochs=ep) for lr in (3e-5, 5e-5) for ep in (6, 12)]
# feat_lr: the annotated-feature weights start from zero and have only ~600 steps to move, so they get
# their own, larger rate (the encoder keeps lr). The default is only a fallback: the feature variants
# tune it (CONFIGS_FEAT), because too low ignores the cues and too high lets them drown the text.
HPARAMS = dict(batch_size=16, max_len=256, warmup=0.1, feat_lr=1e-3)
# 6 settings, not the 12 of a full cross: the encoder rate is fixed at 5e-5 to keep the cost near 1.5x
CONFIGS_FEAT = [dict(lr=5e-5, epochs=ep, feat_lr=fl) for ep in (6, 12) for fl in (1e-4, 1e-3, 1e-2)]


def fit_predict(train_texts, y_train, eval_texts: dict, *, labels, seed, lr, epochs,
                train_marks=None, eval_marks: dict | None = None, train_feats=None,
                eval_feats: dict | None = None, checkpoint=CHECKPOINT, **hp):
    """Fit on ``train_texts``; return class probabilities for each named list of texts.

    ``*_marks`` hold, per text, the character offsets of the tokens to pool at (None = missing,
    which falls back to the first token). The head sees [first token, pooled marks...]. Columns
    follow ``labels``; cross-entropy is class-weighted because the metric is macro-F1.

    ``*_feats`` are (n, d) arrays of annotated features, already scaled. They join the
    head input next to the text vector. That is one linear layer over the concatenation, written as
    two layers whose logits add, so the feature block can have its own learning rate.
    """
    import torch
    from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

    hp = {**HPARAMS, **hp}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tok = AutoTokenizer.from_pretrained(checkpoint)
    enc = AutoModel.from_pretrained(checkpoint, dtype=torch.float32).to(dev)
    n_marks = len(train_marks[0]) if train_marks else 0
    assert not n_marks or eval_marks is not None, "trained with marks but no eval marks"
    head = torch.nn.Sequential(torch.nn.Dropout(0.1),
                               torch.nn.Linear(enc.config.hidden_size * (1 + n_marks), len(labels))).to(dev)
    n_feat = 0 if train_feats is None else train_feats.shape[1]
    assert not n_feat or eval_feats is not None, "trained with features but no eval features"
    feat_head = torch.nn.Linear(max(n_feat, 1), len(labels), bias=False).to(dev)
    torch.nn.init.zeros_(feat_head.weight)
    y = torch.tensor([labels.index(v) for v in y_train])
    counts = torch.bincount(y, minlength=len(labels)).float()
    loss_fn = torch.nn.CrossEntropyLoss(weight=(len(y) / (len(labels) * counts.clamp(min=1))).to(dev))
    n, bs = len(y), hp["batch_size"]
    steps = epochs * ((n + bs - 1) // bs)
    params = list(enc.parameters()) + list(head.parameters())
    groups = [dict(params=params, lr=lr)] + ([dict(params=list(feat_head.parameters()), lr=hp["feat_lr"])] if n_feat else [])
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(hp["warmup"] * steps), steps)

    def logits(texts, marks, feats):
        b = tok(list(texts), padding=True, truncation=True, max_length=hp["max_len"],
                return_offsets_mapping=True, return_tensors="pt")
        off = b.pop("offset_mapping")
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            h = enc(**{k: v.to(dev) for k, v in b.items()}).last_hidden_state
        parts = [h[:, 0]]
        for j in range(n_marks):
            idx = []
            for i in range(len(texts)):
                c = marks[i][j] if marks is not None else None
                hit = [] if c is None else ((off[i, :, 0] <= c) & (off[i, :, 1] > c)).nonzero().flatten().tolist()
                idx.append(hit[0] if hit else 0)
            parts.append(h[torch.arange(len(texts)), torch.tensor(idx, device=dev)])
        out = head(torch.cat(parts, -1).float())
        return out + feat_head(torch.tensor(feats, dtype=torch.float32, device=dev)) if n_feat else out

    for _ in range(epochs):
        enc.train(); head.train()
        order = rng.permutation(n)
        for i in range(0, n, bs):
            ix = order[i:i + bs]
            out = logits([train_texts[j] for j in ix], [train_marks[j] for j in ix] if train_marks else None,
                         train_feats[ix] if n_feat else None)
            loss_fn(out, y[ix].to(dev)).backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad()

    enc.eval(); head.eval()
    probs = {}
    with torch.no_grad():
        for name, texts in eval_texts.items():
            m = eval_marks[name] if eval_marks else None
            f = eval_feats[name] if n_feat else None
            chunks = [logits(texts[i:i + 64], m[i:i + 64] if m else None,
                             f[i:i + 64] if n_feat else None).softmax(-1).cpu().numpy()
                      for i in range(0, len(texts), 64)]
            probs[name] = np.concatenate(chunks) if chunks else np.empty((0, len(labels)))
    return probs


def select_and_predict(train_texts, y_train, groups_train, eval_texts, *, labels, seed,
                       configs=CONFIGS, train_marks=None, eval_marks=None, train_feats=None, eval_feats=None,
                       fit=fit_predict):
    """Pick the config with the best macro-F1 on one grouped inner split of the training rows,
    refit it on all of them, and predict ``eval_texts``. Returns (probs, chosen config)."""
    y_train = np.asarray(y_train)
    inner_fit, inner_val = next(iter(evaluate.folds(y_train, groups_train, seed)))
    take = lambda xs, ix: None if xs is None else [xs[i] for i in ix]  # noqa: E731
    take_feats = lambda xs, ix: None if xs is None else xs[ix]  # noqa: E731
    scores = []
    for cfg in configs:
        p = fit(take(train_texts, inner_fit), y_train[inner_fit], {"val": take(train_texts, inner_val)},
                labels=labels, seed=seed, train_marks=take(train_marks, inner_fit),
                eval_marks=None if train_marks is None else {"val": take(train_marks, inner_val)},
                train_feats=take_feats(train_feats, inner_fit),
                eval_feats=None if train_feats is None else {"val": take_feats(train_feats, inner_val)},
                **cfg)["val"]
        scores.append(evaluate.macro_f1(y_train[inner_val], np.array(labels)[p.argmax(-1)], labels))
    best = configs[int(np.argmax(scores))]
    probs = fit(train_texts, y_train, eval_texts, labels=labels, seed=seed, train_marks=train_marks,
                eval_marks=eval_marks, train_feats=train_feats, eval_feats=eval_feats, **best)
    return probs, {**best, "inner_val_f1": float(max(scores))}

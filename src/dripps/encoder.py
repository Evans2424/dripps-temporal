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
HPARAMS = dict(batch_size=16, max_len=256, warmup=0.1)


def fit_predict(train_texts, y_train, eval_texts: dict, *, labels, seed, lr, epochs,
                train_marks=None, eval_marks: dict | None = None, checkpoint=CHECKPOINT, **hp):
    """Fit on ``train_texts``; return class probabilities for each named list of texts.

    ``*_marks`` hold, per text, the character offsets of the tokens to pool at (None = missing,
    which falls back to the first token). The head sees [first token, pooled marks...]. Columns
    follow ``labels``; cross-entropy is class-weighted because the metric is macro-F1.
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
    y = torch.tensor([labels.index(v) for v in y_train])
    counts = torch.bincount(y, minlength=len(labels)).float()
    loss_fn = torch.nn.CrossEntropyLoss(weight=(len(y) / (len(labels) * counts.clamp(min=1))).to(dev))
    n, bs = len(y), hp["batch_size"]
    steps = epochs * ((n + bs - 1) // bs)
    params = list(enc.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(hp["warmup"] * steps), steps)

    def logits(texts, marks):
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
        return head(torch.cat(parts, -1).float())

    for _ in range(epochs):
        enc.train(); head.train()
        order = rng.permutation(n)
        for i in range(0, n, bs):
            ix = order[i:i + bs]
            out = logits([train_texts[j] for j in ix], [train_marks[j] for j in ix] if train_marks else None)
            loss_fn(out, y[ix].to(dev)).backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); sched.step(); opt.zero_grad()

    enc.eval(); head.eval()
    probs = {}
    with torch.no_grad():
        for name, texts in eval_texts.items():
            m = eval_marks[name] if eval_marks else None
            chunks = [logits(texts[i:i + 64], m[i:i + 64] if m else None).softmax(-1).cpu().numpy()
                      for i in range(0, len(texts), 64)]
            probs[name] = np.concatenate(chunks) if chunks else np.empty((0, len(labels)))
    return probs


def select_and_predict(train_texts, y_train, groups_train, eval_texts, *, labels, seed,
                       configs=CONFIGS, train_marks=None, eval_marks=None, fit=fit_predict):
    """Pick the config with the best macro-F1 on one grouped inner split of the training rows,
    refit it on all of them, and predict ``eval_texts``. Returns (probs, chosen config)."""
    y_train = np.asarray(y_train)
    inner_fit, inner_val = next(iter(evaluate.folds(y_train, groups_train, seed)))
    take = lambda xs, ix: None if xs is None else [xs[i] for i in ix]  # noqa: E731
    scores = []
    for cfg in configs:
        p = fit(take(train_texts, inner_fit), y_train[inner_fit], {"val": take(train_texts, inner_val)},
                labels=labels, seed=seed, train_marks=take(train_marks, inner_fit),
                eval_marks=None if train_marks is None else {"val": take(train_marks, inner_val)},
                **cfg)["val"]
        scores.append(evaluate.macro_f1(y_train[inner_val], np.array(labels)[p.argmax(-1)], labels))
    best = configs[int(np.argmax(scores))]
    probs = fit(train_texts, y_train, eval_texts, labels=labels, seed=seed, train_marks=train_marks,
                eval_marks=eval_marks, **best)
    return probs, {**best, "inner_val_f1": float(max(scores))}

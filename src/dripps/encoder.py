"""Fine-tune a multilingual encoder on the whole sentence (T1).

Plain torch, fixed hyperparameters, no early stopping: nothing here sees the
held-out fold, so there is no tuning on test. ``torch`` and ``transformers`` are
imported inside the function so the rest of the package (and ``make test``)
works without them.
"""

from __future__ import annotations

import numpy as np

CHECKPOINT = "xlm-roberta-base"
HPARAMS = dict(epochs=4, lr=2e-5, batch_size=16, max_len=256, warmup=0.1)


def fit_predict(train_texts, y_train, eval_texts: dict, *, labels, seed, checkpoint=CHECKPOINT, **hp):
    """Fit on ``train_texts``; return class probabilities for each named list of texts.

    Columns follow ``labels``. Unweighted cross-entropy, matching B3.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    hp = {**HPARAMS, **hp}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    tok = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=len(labels), dtype=torch.float32).to(dev)
    y = torch.tensor([labels.index(v) for v in y_train])
    n, bs = len(y), hp["batch_size"]
    steps = hp["epochs"] * ((n + bs - 1) // bs)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(hp["warmup"] * steps), steps)

    def batch(texts):
        return {k: v.to(dev) for k, v in tok(list(texts), padding=True, truncation=True,
                                             max_length=hp["max_len"], return_tensors="pt").items()}

    model.train()
    for _ in range(hp["epochs"]):
        order = rng.permutation(n)
        for i in range(0, n, bs):
            idx = order[i:i + bs]
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                out = model(**batch([train_texts[j] for j in idx]), labels=y[idx].to(dev))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad()

    model.eval()
    probs = {}
    with torch.no_grad():
        for name, texts in eval_texts.items():
            chunks = [model(**batch(texts[i:i + 64])).logits.float().softmax(-1).cpu().numpy()
                      for i in range(0, len(texts), 64)]
            probs[name] = np.concatenate(chunks) if chunks else np.empty((0, len(labels)))
    del model, opt
    torch.cuda.empty_cache()
    return probs

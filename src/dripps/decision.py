"""Jev-style decisions: a closed set of options in, one probability per option out, from a single forward pass.

The model never writes text. The prompt ends where the next token is an option letter, the logits of the
three letters are read and soft-maxed (the mechanism of the open reproductions listed in the Jev Decision
Index: option-logit read on a stock LLM, or one denoise step of a diffusion LM behind ``/v1/systemone``).
Zero-shot, so there is nothing to fit except an optional calibration on the training fold.

An engine is any function ``score(prompts) -> (n, 3)`` of probabilities in option-slot order; ``hf_engine``
and ``score_http`` are the two used here. torch and transformers are imported inside ``hf_engine`` only.
"""

from __future__ import annotations

import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from sklearn.linear_model import LogisticRegression

from . import schema

LABELS = tuple(schema.TR_LABELS)  # sorted, like the columns of every predict_proba in the repo
assert list(LABELS) == sorted(LABELS)
LETTERS = "ABC"
#: order[slot] = label index. The three cyclic shifts put every label in every slot once, which cancels a
#: pure position bias exactly (all six permutations would cost twice as much for the same guarantee).
CYCLES = ((0, 1, 2), (1, 2, 0), (2, 0, 1))
TEMPLATES = {
    1: dict(
        question="The sentence below is Portuguese and contains a participial clause introduced by 'tendo'. "
                 "Relative to the situation of the main clause, when did the situation of the participial clause happen?",
        marks="The participial verb group is marked with [[ ]] and the main verb with << >>.",
        marks_apc="The participial verb group is marked with [[ ]].",
        options={"Ant": "Before the situation of the main clause",
                 "Post": "After the situation of the main clause",
                 "Simul": "At the same time as, or overlapping with, the situation of the main clause"},
        sentence="Sentence", tail="Answer with the letter only."),
    2: dict(  # sensitivity only: the same task worded in Portuguese
        question="A frase seguinte contém uma oração participial introduzida por 'tendo'. Em relação à situação da "
                 "oração principal, quando ocorreu a situação da oração participial?",
        marks="O grupo verbal participial está entre [[ ]] e o verbo principal entre << >>.",
        marks_apc="O grupo verbal participial está entre [[ ]].",
        options={"Ant": "Antes da situação da oração principal",
                 "Post": "Depois da situação da oração principal",
                 "Simul": "Ao mesmo tempo que, ou em sobreposição com, a situação da oração principal"},
        sentence="Frase", tail="Responda apenas com a letra."),
}


def build_prompt(text: str, order=CYCLES[0], template: int = 1, marked: bool = False) -> dict:
    """One decision: the sentence, the question and the three options placed in slots A, B, C by ``order``."""
    t = TEMPLATES[template]
    # the note names only the marks this sentence has: the main verb is found in about two thirds of the rows
    note = (" " + (t["marks"] if "<<" in text else t["marks_apc"])) if marked and "[[" in text else ""
    return {"state": text, "question": t["question"] + note,
            "options": [t["options"][LABELS[i]] for i in order], "sentence": t["sentence"], "tail": t["tail"]}


def render_user(p: dict) -> str:
    opts = "\n".join(f"{L}. {o}" for L, o in zip(LETTERS, p["options"]))
    return f"{p['question']}\n\n{p['sentence']}: {p['state']}\n\n{opts}\n{p['tail']}"


def letter_ids(tok, prefix: str) -> list[int]:
    """Token id of each option letter as it follows ``prefix`` ("...Answer:"): exactly one token each, all different.
    The letter carries its own leading space, so the prefix never ends in one (BPE would merge it into the letter)."""
    enc = lambda s: tok.encode(s, add_special_tokens=False)  # noqa: E731
    base = enc(prefix)
    ids = []
    for L in LETTERS:
        full = enc(prefix + " " + L)
        if full[:len(base)] != base or len(full) != len(base) + 1:
            raise ValueError(f"option letter {L!r} is not a single token after {prefix[-20:]!r}: {full[len(base):]}")
        ids.append(full[-1])
    if len(set(ids)) != len(ids):
        raise ValueError(f"option letters share a token id: {ids}")
    return ids


def per_order(score, texts, **kw) -> np.ndarray:
    """(3, n, 3): the probabilities of the labels, one block per option order, remapped from slots to labels."""
    out = np.zeros((len(CYCLES), len(texts), len(LABELS)))
    for k, order in enumerate(CYCLES):
        S = np.asarray(score([build_prompt(t, order, **kw) for t in texts]))  # slot order
        out[k][:, list(order)] = S  # slot s holds label order[s]
    return out


def debias(P: np.ndarray) -> np.ndarray:
    """Geometric mean over the three option orders, renormalised. A slot preference adds a constant to the logits of
    whichever label sits in that slot, and every label visits every slot once, so the log-probabilities average it
    out exactly (an arithmetic mean of probabilities would not)."""
    L = np.log(np.clip(P, 1e-6, 1.0)).mean(0)
    E = np.exp(L - L.max(-1, keepdims=True))
    return E / E.sum(-1, keepdims=True)


def calibrate(P_tr, y_tr, P_te) -> np.ndarray:
    """Multinomial logit on the three log-probabilities (12 parameters), fitted on training rows only.
    Raw LLM probabilities are model scores, not odds; this is the one place labels touch the decision engine."""
    f = lambda P: np.log(np.clip(P, 1e-6, 1.0))  # noqa: E731
    m = LogisticRegression(max_iter=1000).fit(f(P_tr), y_tr)
    assert list(m.classes_) == list(LABELS), m.classes_
    return m.predict_proba(f(P_te))


def render_chat(tok, p: dict) -> str:
    """The chat-template prompt with thinking off, ending in "Answer:" so the next token is the letter."""
    s = tok.apply_chat_template([{"role": "user", "content": render_user(p)}], tokenize=False,
                                add_generation_prompt=True, enable_thinking=False)
    if "<think>" in s and "</think>" not in s.rsplit("<think>", 1)[1]:  # a template that ignores the flag
        raise ValueError("the chat template left a thinking block open")
    return s + "Answer:"


def hf_engine(model_id: str, batch: int = 8, revision: str | None = None):
    """Load a stock causal LM once; the returned ``score(prompts) -> (n, 3)`` reads the option-letter probabilities
    in one forward pass per batch (left padding, fp32 softmax)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype="auto", device_map="auto")
    except ValueError as e:  # vision-language checkpoints (Qwen3.5 to 3.8) register under another Auto class
        if "Unrecognized configuration class" not in str(e):
            raise
        model = AutoModelForImageTextToText.from_pretrained(model_id, revision=revision, dtype="auto", device_map="auto")
    model.eval()

    def score(prompts: list[dict]) -> np.ndarray:
        if not prompts:
            return np.zeros((0, len(LETTERS)))
        texts = [render_chat(tok, p) for p in prompts]
        ids = letter_ids(tok, texts[0])
        out = np.zeros((len(texts), len(LETTERS)))
        by_len = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        with torch.inference_mode():
            for s in range(0, len(by_len), batch):
                ix = by_len[s:s + batch]
                enc = tok([texts[i] for i in ix], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
                logits = model(**enc, logits_to_keep=1).logits[:, -1, :].float()
                out[ix] = torch.softmax(logits[:, ids], dim=-1).cpu().numpy()
        return out

    return score


def score_http(url: str, prompts: list[dict], model: str | None = None, timeout: int = 300, retries: int = 4,
               workers: int = 4) -> np.ndarray:
    """The same decisions through an open-jev style ``/v1/systemone`` server (e.g. a diffusion LM)."""
    def one(p):
        body = {"state": p["state"], "questions": {"rel": {"type": "choice", "instructions": p["question"],
                "criteria": dict(zip(LETTERS, p["options"]))}}}
        if model:
            body["model"] = model
        req = urllib.request.Request(url, json.dumps(body).encode(), {"Content-Type": "application/json"})
        for attempt in range(retries):  # a busy server answers 503; one failure must not lose a whole text variant
            try:
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    probs = json.load(r)["answers"]["rel"]["probabilities"]
                return [probs[L] for L in LETTERS]
            except OSError:
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)

    with ThreadPoolExecutor(workers) as ex:
        return np.array(list(ex.map(one, prompts)))

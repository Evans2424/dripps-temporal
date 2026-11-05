"""Jev-style decisions: a closed set of options in, one probability per option out, from a single forward pass.

The model never writes text. The prompt ends where the next token is an option letter, the logits of the
three letters are read and soft-maxed (the mechanism of the open reproductions listed in the Jev Decision
Index: option-logit read on a stock LLM, or one denoise step of a diffusion LM behind ``/v1/systemone``).
Zero-shot, so there is nothing to fit except an optional calibration on the training fold.

An engine is any function ``score(prompts) -> (n, 3)`` of probabilities in option-slot order; ``hf_engine``
and ``score_http`` are the two used here. torch and transformers are imported inside ``hf_engine`` only.
"""

from __future__ import annotations

import http.client
import json
import os
import threading
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import FIRST_EXCEPTION, ThreadPoolExecutor
from concurrent.futures import wait as wait_futures

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
        sentence="Sentence", tail="Answer with the letter only.", answer="Answer:"),
    2: dict(  # sensitivity only: the same task worded in Portuguese
        question="A frase seguinte contém uma oração participial introduzida por 'tendo'. Em relação à situação da "
                 "oração principal, quando ocorreu a situação da oração participial?",
        marks="O grupo verbal participial está entre [[ ]] e o verbo principal entre << >>.",
        marks_apc="O grupo verbal participial está entre [[ ]].",
        options={"Ant": "Antes da situação da oração principal",
                 "Post": "Depois da situação da oração principal",
                 "Simul": "Ao mesmo tempo que, ou em sobreposição com, a situação da oração principal"},
        sentence="Frase", tail="Responda apenas com a letra.", answer="Resposta:"),
}


def build_prompt(text: str, order=CYCLES[0], template: int = 1, marked: bool = False) -> dict:
    """One decision: the sentence, the question and the three options placed in slots A, B, C by ``order``."""
    t = TEMPLATES[template]
    # the note names only the marks this sentence has: the main verb is found in about two thirds of the rows
    note = (" " + (t["marks"] if "<<" in text else t["marks_apc"])) if marked and "[[" in text else ""
    return {"state": text, "question": t["question"] + note,
            "options": [t["options"][LABELS[i]] for i in order], "sentence": t["sentence"], "tail": t["tail"],
            "answer": t["answer"]}


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
    """The chat-template prompt with thinking off, ending in the template's answer cue ("Answer:") so the next token
    is the letter."""
    s = tok.apply_chat_template([{"role": "user", "content": render_user(p)}], tokenize=False,
                                add_generation_prompt=True, enable_thinking=False)
    if "<think>" in s and "</think>" not in s.rsplit("<think>", 1)[1]:  # a template that ignores the flag
        raise ValueError("the chat template left a thinking block open")
    return s + p["answer"]


def render_plain(p: dict) -> str:
    """For checkpoints without a chat template (Gervásio, Sabiá): the same user text as a plain completion."""
    return render_user(p) + "\n" + p["answer"]


def hf_engine(model_id: str, batch: int = 8, revision: str | None = None):
    """Load a stock causal LM once; the returned ``score(prompts) -> (n, 3)`` reads the option-letter probabilities
    in one forward pass per batch (left padding, fp32 softmax). ``score.format`` is "chat" or "plain" (no chat
    template), and ``score.nll(texts)`` is each raw sentence's mean token negative log-likelihood (no prompt)."""
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
        texts = [render_chat(tok, p) if fmt == "chat" else render_plain(p) for p in prompts]
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

    def nll(sentences: list[str]) -> np.ndarray:
        """Mean token NLL of each sentence on its own: how familiar its variety's text is to the model."""
        out = np.zeros(len(sentences))
        bos = [tok.bos_token_id] if tok.bos_token_id is not None else []
        with torch.inference_mode():
            for s in range(0, len(sentences), batch):
                ids = [bos + tok.encode(x, add_special_tokens=False) for x in sentences[s:s + batch]]
                n = max(map(len, ids))
                x = torch.tensor([i + [tok.pad_token_id] * (n - len(i)) for i in ids], device=model.device)
                m = torch.tensor([[1] * len(i) + [0] * (n - len(i)) for i in ids], device=model.device)
                lp = torch.log_softmax(model(input_ids=x, attention_mask=m).logits[:, :-1].float(), -1)
                tl = lp.gather(-1, x[:, 1:, None])[..., 0] * m[:, 1:]
                out[s:s + len(ids)] = (-tl.sum(1) / m[:, 1:].sum(1)).cpu().numpy()
        return out

    fmt = "chat" if tok.chat_template else "plain"
    score.format, score.nll = fmt, nll
    return score


#: Jev reads the option names as well as their descriptions, and positional names (A, B, C) carry no meaning and
#: can bias it (docs.typesafe.ai, Choice), so the HTTP engine names each option by its reading. The slot order is
#: still the order of the keys, so the three cyclic orders keep cancelling position bias.
OPTION_NAMES = {"Ant": "before", "Post": "after", "Simul": "same_time"}
_TEXT_LABEL = {text: lab for t in TEMPLATES.values() for lab, text in t["options"].items()}


TYPESAFE_ORIGIN = "https://api.typesafe.ai"  # the only place the key is ever sent
#: busy or overloaded (Jev: 429, 529; proxies: 502-504, Cloudflare 520-524): worth retrying. Any other status is final.
RETRY_STATUS = {408, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524, 529}
MAX_WAIT = 600  # seconds; a Retry-After above this fails the prompt instead of hammering a rate limit


def is_typesafe(url: str) -> bool:
    """Same host and port as TYPESAFE_ORIGIN (case-insensitive, default ports filled in). The one place the decision
    is made, so 10_decision.py and score_http can't disagree about where the key goes."""
    u, o = urllib.parse.urlsplit(url), urllib.parse.urlsplit(TYPESAFE_ORIGIN)
    port = lambda x: x.port or {"https": 443, "http": 80}.get(x.scheme)  # noqa: E731
    return u.hostname == o.hostname and u.scheme == o.scheme and port(u) == port(o)


def http_body(p: dict, model: str | None) -> tuple[list[str], dict]:
    """What the HTTP engine sends for one prompt, and the option names in slot order. Everything that defines the
    request is here, so 10_decision.py can fingerprint it for the cache."""
    names = [OPTION_NAMES[_TEXT_LABEL[o]] for o in p["options"]]  # slot order
    body = {"state": p["state"], "questions": {"rel": {"type": "choice", "instructions": p["question"],
            "criteria": dict(zip(names, p["options"]))}}}
    if model:
        body["model"] = model
    return names, body


def _retry_after(e: urllib.error.HTTPError) -> float:
    try:
        return float(e.headers.get("Retry-After", 0))
    except (TypeError, ValueError):  # an HTTP-date: fall back to the backoff
        return 0.0


def _pause(seconds: float, stop: threading.Event) -> None:
    stop.wait(seconds)  # wakes at once when another prompt has failed


def score_http(url: str, prompts: list[dict], model: str | None = None, timeout: int = 300, retries: int = 8,
               workers: int = 4, seen: dict | None = None) -> np.ndarray:
    """The same decisions through a ``/v1/systemone`` server: TypeSafe's Jev API (key in ``TYPESAFE_API_KEY``,
    docs.typesafe.ai/api/system-one) or an open-jev style one (e.g. OpenJev's shim, a diffusion LM; no key).

    The key goes only to api.typesafe.ai over https, never to another --url. Busy answers (RETRY_STATUS), dropped
    connections and truncated bodies are retried with backoff (at least Retry-After, at most MAX_WAIT). Any other
    status, a reply without the expected options, or an answer from a model other than the pinned one fails at
    once. The first failure, from any worker, stops every other prompt: queued ones are never sent, retrying ones
    give up at their next attempt, so nothing more is sent or billed.

    TypeSafe must answer with exactly ``model``. Another server answers with its own fingerprint
    ("<dir> T=... shim=<file>@<sha>"): the first one seen is kept in ``seen`` (one dict for a whole run, seeded from
    the cache's meta.json on resume) and every later answer must match it."""
    u = urllib.parse.urlsplit(url)
    if (u.hostname or "") == "api.typesafe.ai" and u.scheme != "https":
        raise ValueError(f"{url}: TypeSafe must be called over https")
    typesafe = is_typesafe(url)
    headers = {"Content-Type": "application/json"}
    if typesafe:
        if not os.environ.get("TYPESAFE_API_KEY"):
            raise RuntimeError("TYPESAFE_API_KEY is not set (export it, or put it in .env)")
        if not model or model.endswith(("-latest", "-preview")):
            raise ValueError(f"pin a versioned Jev id such as jev-1.14.0, not {model!r}: an alias can move mid-run")
        headers["Authorization"] = f"Bearer {os.environ['TYPESAFE_API_KEY']}"
    seen = {} if seen is None else seen
    stop = threading.Event()

    def one(p):
        names, body = http_body(p, model)
        req = urllib.request.Request(url, json.dumps(body, ensure_ascii=False).encode(), headers)
        for attempt in range(retries):  # one transient failure must not lose a whole text variant
            if stop.is_set():
                raise RuntimeError("stopped: another prompt failed")
            wait = 2 ** attempt
            try:
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    raw = r.read()
                out = json.loads(raw)
            except urllib.error.HTTPError as e:  # before OSError: HTTPError is one
                if e.code not in RETRY_STATUS:
                    raise RuntimeError(f"{e.code} from {url}: {e.read()[:300]!r}") from e
                err, wait = e, max(wait, _retry_after(e))
                if wait > MAX_WAIT:
                    raise RuntimeError(f"{e.code} from {url}: asked to wait {wait:.0f}s (> {MAX_WAIT}s)") from e
            except (OSError, json.JSONDecodeError, http.client.HTTPException) as e:  # dropped or truncated: retry
                err = e
            else:
                try:  # a complete reply in the wrong shape will not fix itself: fail now
                    probs = out["answers"]["rel"]["probabilities"]
                    vals = [float(probs[n]) for n in names]
                except (KeyError, TypeError, ValueError) as e:
                    raise RuntimeError(f"unexpected reply from {url}: {raw[:300]!r}") from e
                answered = out.get("model")
                if typesafe and answered != model:  # an alias or upgrade mixes versions
                    raise RuntimeError(f"asked for {model}, answered by {answered}")
                if not typesafe and answered is not None:
                    first = seen.setdefault("model", answered)  # atomic in CPython: the first thread's answer wins
                    if answered != first:
                        raise RuntimeError(f"server changed model mid-run: {first!r} -> {answered!r}")
                return vals
            if attempt == retries - 1:
                raise RuntimeError(f"{url}: still failing after {retries} attempts: {err!r}") from err
            _pause(min(wait, MAX_WAIT), stop)

    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(one, p) for p in prompts]
        done, _ = wait_futures(futs, return_when=FIRST_EXCEPTION)  # the first failure from any worker, not in order
        failed = next((f for f in futs if f in done and f.exception() is not None), None)
        if failed is not None:
            stop.set()
            ex.shutdown(cancel_futures=True)  # queued requests would only be billed and thrown away
            raise failed.exception()
        return np.array([f.result() for f in futs])

"""Torch-free checks for the Jev-style decision step: option orders, debiasing, calibration, token guards."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dripps import decision  # noqa: E402

TEXTS = ["Saiu, tendo comido.", "Chegou, tendo dormido.", "Falou, tendo bebido.", "Ficou, tendo lido."]


def slot_of(prompt, label):
    """Slot (0, 1, 2) in which the prompt put the option for ``label``."""
    return [o for o in prompt["options"]].index(decision.TEMPLATES[1]["options"][label])


def test_every_label_visits_every_slot_once():
    for lab in range(3):
        assert sorted(decision.CYCLES[k].index(lab) for k in range(3)) == [0, 1, 2]


def test_pure_position_bias_cancels_after_debiasing():
    always_a = lambda prompts: np.tile([0.8, 0.1, 0.1], (len(prompts), 1))  # noqa: E731  (prefers slot A)
    P = decision.debias(decision.per_order(always_a, TEXTS))
    assert np.allclose(P, 1 / 3)  # (0.8 + 0.1 + 0.1) / 3 for every label: no label preferred


def test_a_slot_preference_in_logit_space_cancels_exactly():
    z = np.array([2.0, 0.0, -1.0])  # the content: Ant is the clear answer
    bias = np.array([1.5, 0.0, -1.0])  # and the model likes slot A, dislikes slot C

    def biased(prompts):
        out = []
        for p in prompts:
            slots = [decision.LABELS.index(next(k for k, v in decision.TEMPLATES[1]["options"].items() if v == o)) for o in p["options"]]
            e = np.exp(z[slots] + bias)
            out.append(e / e.sum())
        return np.array(out)

    P = decision.debias(decision.per_order(biased, TEXTS))
    want = np.exp(z) / np.exp(z).sum()
    assert np.allclose(P, want)


def test_a_scorer_that_knows_the_label_is_mapped_back():
    truth = {t: i for i, t in enumerate(TEXTS)}  # row i -> label i % 3 below

    def oracle(prompts):
        out = np.full((len(prompts), 3), 0.05)
        for r, p in enumerate(prompts):
            lab = truth[p["state"]] % 3
            out[r, slot_of(p, decision.LABELS[lab])] = 0.9
        return out

    P = decision.debias(decision.per_order(oracle, TEXTS))
    assert list(P.argmax(1)) == [i % 3 for i in range(len(TEXTS))]
    assert np.allclose(P.sum(1), 1.0)


def test_calibration_uses_training_rows_only():
    rng = np.random.default_rng(0)
    P_tr = rng.dirichlet([2, 2, 2], 90)
    y_tr = np.array(decision.LABELS)[P_tr.argmax(1)]
    P_te = rng.dirichlet([2, 2, 2], 10)
    a = decision.calibrate(P_tr, y_tr, P_te)
    b = decision.calibrate(P_tr, y_tr, P_te + 0.0)  # no test labels exist to poison; the output is a pure function of train
    assert np.allclose(a, b) and np.allclose(a.sum(1), 1) and a.shape == (10, 3)
    y_bad = np.roll(y_tr, 1)  # labels of other rows: a different fit, a different answer
    assert not np.allclose(decision.calibrate(P_tr, y_bad, P_te), a)


class Tok:
    """Whitespace tokenizer; ``multi`` makes one letter split into two tokens, ``same`` makes two letters collide."""
    def __init__(self, multi=None, same=False):
        self.multi, self.same = multi, same

    def encode(self, s, add_special_tokens=False):
        out = []
        for w in s.split():
            if w == self.multi:
                out += [hash(w) % 997, 5000]
            elif self.same and w in "BC":
                out.append(4242)
            else:
                out.append(sum(map(ord, w)))
        return out


def test_letters_are_single_distinct_tokens():
    assert len(set(decision.letter_ids(Tok(), "user ...Answer:"))) == 3
    with pytest.raises(ValueError, match="single token"):
        decision.letter_ids(Tok(multi="B"), "x Answer:")
    with pytest.raises(ValueError, match="share"):
        decision.letter_ids(Tok(same=True), "x Answer:")


def test_render_chat_rejects_an_open_thinking_block():
    class T:
        def __init__(self, tail):
            self.tail = tail

        def apply_chat_template(self, msgs, tokenize, add_generation_prompt, enable_thinking):
            assert not enable_thinking
            return "<|user|>" + msgs[0]["content"] + self.tail

    p = decision.build_prompt(TEXTS[0])
    assert decision.render_chat(T("<|assistant|><think>\n\n</think>\n\n"), p).endswith("Answer:")
    with pytest.raises(ValueError, match="thinking"):
        decision.render_chat(T("<|assistant|><think>\n"), p)


def test_prompt_keeps_the_option_order_and_marks_note():
    p = decision.build_prompt("X, [[ tendo ido ]] << saiu >>.", order=(2, 0, 1), marked=True)
    assert p["options"][0].startswith("At the same time") and p["options"][1].startswith("Before")
    assert "<< >>" in p["question"] and "[[ ]]" not in decision.build_prompt("X", marked=False)["question"]
    only_apc = decision.build_prompt("X, [[ tendo ido ]] saiu.", marked=True)["question"]
    assert "[[ ]]" in only_apc and "<< >>" not in only_apc  # no note about a main-verb mark the sentence lacks
    assert decision.build_prompt("X, tendo ido.", marked=True)["question"] == decision.build_prompt("X, tendo ido.")["question"]
    assert "A. At the same time" in decision.render_user(p)


def test_http_engine_sends_one_choice_question_and_reads_the_probabilities():
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    seen = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body)
            a = 0.1 * (1 + TEXTS.index(body["state"]))  # the reply identifies the sentence, so row order can be checked
            reply = {"answers": {"rel": {"probabilities": {"A": a, "B": 1 - a - 0.05, "C": 0.05}}}}
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps(reply).encode())

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        out = decision.score_http(f"http://127.0.0.1:{srv.server_port}/v1/systemone",
                                  [decision.build_prompt(t) for t in TEXTS[:2]], model="m")
    finally:
        srv.shutdown()
    assert np.allclose(out, [[0.1, 0.85, 0.05], [0.2, 0.75, 0.05]])  # requests run concurrently; rows keep the input order
    q = seen[0]["questions"]["rel"]
    assert {b["state"] for b in seen} == set(TEXTS[:2]) and all(b["model"] == "m" for b in seen) and q["type"] == "choice"
    assert list(q["criteria"]) == ["A", "B", "C"] and q["criteria"]["A"].startswith("Before")

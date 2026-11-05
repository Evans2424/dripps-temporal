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
            names = list(body["questions"]["rel"]["criteria"])  # slot order
            reply = {"answers": {"rel": {"probabilities": dict(zip(names, [a, 1 - a - 0.05, 0.05]))}}}
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
    assert list(q["criteria"]) == ["before", "after", "same_time"] and q["criteria"]["before"].startswith("Before")
    rotated = decision.build_prompt(TEXTS[0], order=decision.CYCLES[1])  # names follow the options, not the slots
    assert decision.OPTION_NAMES[decision._TEXT_LABEL[rotated["options"][0]]] == "after"


def test_http_engine_talks_to_the_jev_api(monkeypatch):
    """Key only to TypeSafe; busy and malformed replies retried (Retry-After honoured); 401, a version mismatch, a
    missing model field and an alias fail at once; the first failure stops the queued requests."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    script = []  # what the next requests get: a status code, "bad" (a 200 with a broken body) or a model id
    seen = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            seen.append(self.headers.get("Authorization"))
            step = script.pop(0) if script else "jev-1.14.0"
            if isinstance(step, int):
                self.send_response(step)
                if step == 429:
                    self.send_header("Retry-After", "7")
                self.end_headers()
                return
            self.send_response(200)
            self.end_headers()
            reply = {"answers": {"rel": {"probabilities": {"before": 0.2, "after": 0.3, "same_time": 0.5}}}}
            if step != "no-model":
                reply["model"] = step
            self.wfile.write(b"<html>busy" if step == "bad" else json.dumps(reply).encode())

        def log_message(self, *a):
            pass

    slept = []
    monkeypatch.setattr(decision, "_pause", lambda s, stop: slept.append(s))
    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_port}/v1/systemone"
    ps = [decision.build_prompt(TEXTS[0])]
    run = lambda prompts=ps, model="jev-1.14.0", **kw: decision.score_http(url, prompts, model=model, **kw)  # noqa: E731
    try:
        monkeypatch.setenv("TYPESAFE_API_KEY", "k")
        assert np.allclose(run(), [[0.2, 0.3, 0.5]]) and seen[-1] is None  # not TypeSafe: the key is never sent

        monkeypatch.setattr(decision, "TYPESAFE_ORIGIN", url.rsplit("/v1", 1)[0])  # this server now plays TypeSafe
        script[:] = [429, 504, "bad"]
        assert np.allclose(run(), [[0.2, 0.3, 0.5]]) and seen[-1] == "Bearer k"
        assert slept == [7, 2, 4]  # Retry-After over the 1 s backoff, then the backoff
        for step, err in ((401, "401"), ("jev-1.15.0", "answered by jev-1.15.0"), ("no-model", "answered by None")):
            script[:] = [step]
            with pytest.raises(RuntimeError, match=err):
                run()
        with pytest.raises(ValueError, match="pin a versioned"):
            run(model="jev-latest")
        monkeypatch.delenv("TYPESAFE_API_KEY")
        with pytest.raises(RuntimeError, match="TYPESAFE_API_KEY"):
            run()
        monkeypatch.setenv("TYPESAFE_API_KEY", "k")

        n = len(seen)
        script[:] = [401]
        with pytest.raises(RuntimeError, match="401"):
            run([decision.build_prompt(t) for t in TEXTS] * 20, workers=1)
        assert len(seen) <= n + 2  # of 100 queued: the failing one and at most the one already in flight
    finally:
        srv.shutdown()
    with pytest.raises(ValueError, match="https"):
        decision.score_http("http://api.typesafe.ai/v1/systemone", ps, model="jev-1.14.0")


def test_http_engine_pins_an_open_servers_fingerprint(monkeypatch):
    """An open server (OpenJev's shim) answers with its own fingerprint, not the requested id: the run keeps the
    first one, across calls, and stops if a later answer comes from a server with other settings."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    fp = {"v": "openjev-FP8 T=0.85 shim=shim.py@abc123"}
    auth = []

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            auth.append(self.headers.get("Authorization"))  # checked in the test thread, not here
            names = list(body["questions"]["rel"]["criteria"])
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"model": fp["v"], "answers": {"rel": {
                "probabilities": dict(zip(names, [0.6, 0.3, 0.1]))}}}).encode())

        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"model_dir": "openjev-FP8", "T": 0.85, "shim_sha256": "abc123", "extra": 1}).encode())

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_port}/v1/systemone"
    ps = [decision.build_prompt(t) for t in TEXTS]
    try:
        monkeypatch.setenv("TYPESAFE_API_KEY", "k")  # set, to show it is not sent
        seen = {}
        assert decision.score_http(url, ps, model="qwen", seen=seen).shape == (len(TEXTS), 3)
        assert decision.score_http(url, ps, model="qwen", seen=seen).shape == (len(TEXTS), 3)  # same server: fine
        assert seen["model"] == fp["v"] and set(auth) == {None}  # the TypeSafe key never reached this server
        fp["v"] = "openjev-FP8 T=1.1 shim=shim.py@abc123"  # restarted with another readout temperature
        with pytest.raises(RuntimeError, match="changed model mid-run"):
            decision.score_http(url, ps, model="qwen", seen=seen)

        import importlib.util
        spec = importlib.util.spec_from_file_location("d10", Path(__file__).resolve().parents[1] / "experiments/10_decision.py")
        d10 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(d10)
        assert d10.server_version(url) == {"model_dir": "openjev-FP8", "T": 0.85, "noul_t": None, "noul_bias": None,
                                           "flags": None, "shim_sha256": "abc123"}
        with pytest.raises(RuntimeError, match="did not answer"):  # unreachable is an error, never "no version"
            d10.server_version("http://127.0.0.1:1/v1/systemone", tries=2, pause=0)
    finally:
        srv.shutdown()


def test_http_engine_stops_every_worker_at_the_first_failure(monkeypatch):
    """4 workers: prompt 0 keeps getting 503 (retrying) while prompt 1 gets a 401. The 401 must surface, and the
    hundreds of queued prompts must not be sent (or billed)."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    hits = []
    first, second = decision.build_prompt(TEXTS[0])["state"], decision.build_prompt(TEXTS[1])["state"]

    class H(BaseHTTPRequestHandler):
        def do_POST(self):
            st = json.loads(self.rfile.read(int(self.headers["Content-Length"])))["state"]
            hits.append(st)
            code = 503 if st == first else 401 if st == second else 200
            self.send_response(code)
            self.end_headers()
            if code == 200:
                names = ["before", "after", "same_time"]
                self.wfile.write(json.dumps({"answers": {"rel": {"probabilities": dict(zip(names, [.2, .3, .5]))}}}).encode())

        def log_message(self, *a):
            pass

    monkeypatch.setattr(decision, "_pause", lambda s, stop: stop.wait(0.05))  # short real waits, still interruptible
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    ps = [decision.build_prompt(TEXTS[0]), decision.build_prompt(TEXTS[1])] + [decision.build_prompt(TEXTS[2])] * 400
    try:
        with pytest.raises(RuntimeError, match="401"):
            decision.score_http(f"http://127.0.0.1:{srv.server_port}/v1/systemone", ps, model="m", workers=4)
    finally:
        srv.shutdown()
    assert len(hits) < 60, len(hits)  # without the stop, all ~400 queued prompts were sent


def test_typesafe_is_recognised_by_host_not_by_prefix():
    assert decision.is_typesafe("https://API.typesafe.ai:443/v1/systemone")
    assert not decision.is_typesafe("https://api.typesafe.ai.example.com/v1/systemone")
    assert not decision.is_typesafe("http://127.0.0.1:3000/v1/systemone")


def test_template_one_rendering_is_unchanged_by_the_answer_cue():
    """The answer cue moved into the templates; template-1 prompts (every cached hf model) must render as before."""
    class T:
        def apply_chat_template(self, msgs, tokenize, add_generation_prompt, enable_thinking):
            return "<u>" + msgs[0]["content"] + "<a>"

    p = decision.build_prompt(TEXTS[0])
    assert decision.render_chat(T(), p) == "<u>" + decision.render_user(p) + "<a>Answer:"


def test_plain_prompt_for_checkpoints_without_a_chat_template():
    p = decision.build_prompt(TEXTS[0], template=2)
    s = decision.render_plain(p)
    assert s.startswith(decision.TEMPLATES[2]["question"]) and s.endswith("Responda apenas com a letra.\nResposta:")
    assert "A. " in s and TEXTS[0] in s

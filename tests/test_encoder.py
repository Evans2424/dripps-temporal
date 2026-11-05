"""Torch-free checks for the encoder step: text perturbations and fold parity with B3."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dripps import ablate, evaluate  # noqa: E402

T = "Orlando desfilou pela cidade, tendo apostado no contacto interpessoal."


def test_apc_span_picks_the_content_participle():
    (s, e), (ps, pe) = ablate.apc_span(T)
    assert T[ps:pe] == "apostado" and T[s:e] == "tendo apostado"
    sido = "Cláudia saiu, tendo sido nomeada pela prefeita."
    assert sido[slice(*ablate.apc_span(sido)[1])] == "nomeada"


def test_ambiguous_or_missing_span_returns_none():
    assert ablate.apc_span("Tendo saído, tendo chegado, falou.") is None  # two APCs
    assert ablate.apc_span("Ele tendo de ir, saiu.") is None  # no participle
    assert ablate.apc_span("Saiu, tendo os dados sido recolhidos.") is None  # subject between: ambiguous
    assert ablate.apc_span("Saiu, tendo sido eleita a candidata.") is not None
    assert ablate.apc_span("Saiu, tendo sido campeão.") is not None  # copular: 'sido' is the participle
    assert ablate.apc_span("Saiu, tendo o alerta sido dado.") is None  # subject between
    assert ablate.move_apc_initial("Ele saiu, tendo dito que sim, e voltou.") is None


def test_perturbations():
    assert ablate.mask_spans(T, [ablate.apc_span(T)[1]], "<mask>").endswith("tendo <mask> no contacto interpessoal.")
    assert ablate.move_apc_initial(T) == "Tendo apostado no contacto interpessoal, Orlando desfilou pela cidade."
    assert ablate.move_apc_initial("A PSP saiu, tendo dito que sim.", lower_first=True) == "Tendo dito que sim, a PSP saiu."
    assert ablate.move_apc_initial("Subiu ( 61 % ), tendo caído.") == "Tendo caído, Subiu ( 61 % )."
    assert ablate.move_apc_initial("Disse “ sim, tendo saído ”.") is None  # quote split across the clauses
    adv = ablate.adverbial_spans("Saiu dois dias depois, tendo chegado já tarde.")
    assert len(adv) == 2
    out = ablate.mask_random(T, [ablate.apc_span(T)[0]], 2, "<mask>", np.random.default_rng(0))
    assert out.count("<mask>") == 2 and "tendo apostado" in out


def test_encoder_folds_are_the_ones_cv_predict_uses():
    y = np.array(["Ant", "Post", "Simul"] * 40)
    g = np.arange(120) // 2
    seen = []

    class Spy(ClassifierMixin, BaseEstimator):
        def fit(self, X, y):
            self.classes_ = np.unique(y)
            return self

        def predict(self, X):
            seen.append(X[:, 0].copy())
            return np.array(["Ant"] * len(X))

    X = np.arange(120, dtype=float)[:, None]
    evaluate.cv_predict(Spy(), X, y, g, n_repeats=1, seed=7)
    assert [t.astype(int).tolist() for t in seen] == [te.tolist() for _, te in evaluate.folds(y, g, 7)]


def test_mark_shifts_offsets_to_the_marked_text():
    t = "Ele saiu, tendo dito que sim."
    group, verb = (10, 20), (0, 3)
    marked, shift = ablate.mark(t, [(group, "[[ ", " ]]"), (verb, "<< ", " >>")])
    assert marked == "<< Ele >> saiu, [[ tendo dito ]] que sim."
    assert marked[shift(t.index("dito")):][:4] == "dito" and marked[shift(0):][:3] == "Ele"
    assert ablate.mark(t, [])[0] == t


def test_select_and_predict_tunes_only_on_the_training_rows():
    from dripps import encoder

    labels = ["Ant", "Post", "Simul"]
    y = np.array(labels * 40)
    g = np.arange(120) // 2
    texts = [str(i) for i in range(120)]
    train, test = next(iter(evaluate.folds(y, g, 3)))
    calls = []

    def fake_fit(tr_texts, y_tr, ev, *, labels, seed, lr, epochs, train_marks=None, eval_marks=None,
                 train_feats=None, eval_feats=None):
        calls.append((set(tr_texts), set(next(iter(ev.values()))), lr))
        out = {}
        for k, ts in ev.items():  # lr 5e-5 "learns" (returns the true label); 3e-5 always says Ant
            p = np.full((len(ts), 3), 0.1)
            for row, t in enumerate(ts):
                p[row, labels.index(y[int(t)]) if lr == 5e-5 else 0] = 0.8
            out[k] = p
        return out

    cfgs = [dict(lr=3e-5, epochs=1), dict(lr=5e-5, epochs=1)]
    probs, cfg = encoder.select_and_predict([texts[i] for i in train], y[train], g[train],
                                            {"orig": [texts[i] for i in test]}, labels=labels, seed=3,
                                            configs=cfgs, fit=fake_fit)
    held = {texts[i] for i in test}
    assert all(not (tr | ev) & held for tr, ev, _ in calls[:2])         # selection never sees the held-out fold
    assert not calls[0][0] & calls[0][1]                                 # inner fit/validation rows are disjoint
    assert not {g[int(t)] for t in calls[0][0]} & {g[int(t)] for t in calls[0][1]}  # and so are their groups
    assert cfg["lr"] == 5e-5 and len(calls) == 3 and probs["orig"].shape == (len(test), 3)


def test_features_follow_their_rows_through_selection():
    from dripps import encoder

    labels = ["Ant", "Post", "Simul"]
    y = np.array(labels * 40)
    g = np.arange(120) // 2
    texts = [str(i) for i in range(120)]
    feats = np.arange(120, dtype=float)[:, None]  # a feature that is the row index
    train, test = next(iter(evaluate.folds(y, g, 3)))
    seen = []

    def fake_fit(tr_texts, y_tr, ev, *, labels, seed, lr, epochs, train_marks=None, eval_marks=None,
                 train_feats=None, eval_feats=None):
        seen.append((tr_texts, train_feats, next(iter(ev.values())), next(iter(eval_feats.values()))))
        return {k: np.full((len(t), 3), 1 / 3) for k, t in ev.items()}

    encoder.select_and_predict([texts[i] for i in train], y[train], g[train], {"orig": [texts[i] for i in test]},
                               labels=labels, seed=3, configs=[dict(lr=3e-5, epochs=1)], fit=fake_fit,
                               train_feats=feats[train], eval_feats={"orig": feats[test]})
    assert all([int(t) for t in tr] == f[:, 0].tolist() and [int(t) for t in ev] == ef[:, 0].tolist()
               for tr, f, ev, ef in seen)  # inner fit, inner validation and the final refit all stay aligned
    assert len(seen) == 2


def test_cache_keys_are_per_variant(monkeypatch):
    from dripps import encoder

    kw = dict(texts=["a", "b"], marked=["[[a]]", "b"], perturbed={"mask": ["x", None]}, X=np.zeros((2, 3)),
              labels=["Ant", "Post"], seed=1, library="1")
    before = {v: encoder.cache_key(v, **kw) for v in ("plain", "marked", "plain_feat", "marked_feat")}
    monkeypatch.setattr(encoder, "CONFIGS_FEAT", encoder.CONFIGS_FEAT[:3])
    monkeypatch.setitem(encoder.HPARAMS, "feat_lr", 5e-3)
    after = {v: encoder.cache_key(v, **{**kw, "X": np.ones((2, 3))}) for v in before}
    assert after["plain"] == before["plain"] and after["marked"] == before["marked"]  # text-only folds survive
    assert after["plain_feat"] != before["plain_feat"] and after["marked_feat"] != before["marked_feat"]
    assert len(set(before.values())) == 4


def test_earlier_cache_keys_are_unchanged_and_new_ones_differ():
    from dripps import encoder

    kw = dict(texts=["a", "b"], marked=["[[a]]", "b"], perturbed={"mask": ["x", None]},
              X=np.arange(6.0).reshape(2, 3), labels=["Ant", "Post"], seed=1, library="1")
    golden = {"plain": "plain_a9061a1bdd", "marked": "marked_b75d8e7d10",
              "plain_feat": "plain_feat_c86dd33526", "marked_feat": "marked_feat_000915efed"}
    assert {v: encoder.cache_key(v, **kw) for v in golden} == golden  # the lab's cached folds stay valid
    keys = {encoder.cache_key(v, **kw, extra=[1]) for v in ("plain_res", "plain_aux", "marked_res", "plain_txt",
                                                            "plain_res@bertimbau", "plain@bertimbau")}
    assert len(keys) == 6
    assert encoder.cache_key("plain@bertimbau", **kw) != encoder.cache_key("plain@albertina", **kw)
    assert encoder.cache_key("plain_aux", **kw, extra=[1]) != encoder.cache_key("plain_aux", **kw, extra=[2])


def test_variant_names_round_trip():
    from dripps import encoder

    for v, parts in {"plain": ("plain", "", "xlmr"), "marked_feat": ("marked", "feat", "xlmr"),
                     "plain_res@bertimbau": ("plain", "res", "bertimbau"), "marked@albertina": ("marked", "", "albertina")}.items():
        assert encoder.parse(v) == parts and encoder.name(*parts) == v


def test_offsets_and_auxiliary_targets_follow_their_rows_through_selection():
    from dripps import encoder

    labels = ["Ant", "Post", "Simul"]
    y = np.array(labels * 40)
    g = np.arange(120) // 2
    texts = [str(i) for i in range(120)]
    idx = np.arange(120, dtype=float)
    off, aux = np.repeat(idx[:, None], 3, 1), idx[:, None]
    train, test = next(iter(evaluate.folds(y, g, 3)))
    seen = []

    def fake_fit(tr, y_tr, ev, *, labels, seed, lr, epochs, train_marks=None, eval_marks=None, train_feats=None,
                 eval_feats=None, train_off=None, eval_off=None, train_aux=None, aux_sizes=()):
        seen.append(all(int(t) == o[0] == a[0] for t, o, a in zip(tr, train_off, train_aux)) and
                    all(int(t) == o[0] for t, o in zip(next(iter(ev.values())), next(iter(eval_off.values())))))
        return {k: np.full((len(t), 3), 1 / 3) for k, t in ev.items()}

    encoder.select_and_predict([texts[i] for i in train], y[train], g[train], {"orig": [texts[i] for i in test]},
                               labels=labels, seed=3, configs=[dict(lr=3e-5, epochs=1)], fit=fake_fit,
                               train_off=off[train], eval_off={"orig": off[test]}, train_aux=aux[train], aux_sizes=[1])
    assert seen == [True, True]


def test_fit_predict_with_offsets_and_auxiliary_heads_runs():
    import pytest
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from dripps import encoder

    labels = ["Ant", "Post", "Simul"]
    texts = [f"ele saiu, tendo feito {w}." for w in "abcdefgh"]
    y = np.array(labels * 3)[:8]
    off = np.zeros((8, 3)); off[:, 0] = 5.0  # a strong prior for Ant that the zero-initialised head keeps at first
    aux = np.array([[0, 1], [1, -1]] * 4)
    try:
        p = encoder.fit_predict(texts, y, {"orig": texts[:3]}, labels=labels, seed=0, lr=1e-9, epochs=1,
                                train_off=off, eval_off={"orig": off[:3]}, train_aux=aux, aux_sizes=[2, 2], warmup=0.0,
                                checkpoint="hf-internal-testing/tiny-xlm-roberta", batch_size=4)
    except OSError:
        pytest.skip("tiny checkpoint not available offline")
    assert p["orig"].shape == (3, 3) and np.allclose(p["orig"].sum(1), 1)
    assert np.allclose(p["orig"][:, 0], np.exp(5) / (np.exp(5) + 2), atol=1e-3)  # the head starts at the offset


def test_fit_predict_with_features_runs_and_checks_its_inputs():
    import pytest
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from dripps import encoder

    labels = ["Ant", "Post", "Simul"]
    texts = [f"ele saiu, tendo feito {w}." for w in "abcdefgh"]
    y = np.array(labels * 3)[:8]
    feats = np.random.default_rng(0).normal(size=(8, 4))
    try:
        p = encoder.fit_predict(texts, y, {"orig": texts[:3]}, labels=labels, seed=0, lr=5e-5, epochs=1,
                                train_feats=feats, eval_feats={"orig": feats[:3]}, feat_lr=1e-2,
                                checkpoint="hf-internal-testing/tiny-xlm-roberta", batch_size=4)
    except OSError:
        pytest.skip("tiny checkpoint not available offline")
    assert p["orig"].shape == (3, 3) and np.allclose(p["orig"].sum(1), 1)
    with pytest.raises(AssertionError):  # a feature variant must score every eval set with features
        encoder.fit_predict(texts, y, {"orig": texts[:3], "other": texts[:2]}, labels=labels, seed=0, lr=5e-5,
                            epochs=1, train_feats=feats, eval_feats={"orig": feats[:3]},
                            checkpoint="hf-internal-testing/tiny-xlm-roberta", batch_size=4)


def test_methods_doc_names_the_tagger_the_script_loads():
    root = Path(__file__).resolve().parents[1]
    model = "pt_core_news_sm"
    assert model in (root / "experiments/08_encoder.py").read_text()
    methods = (root / "docs/methods.md").read_text()
    assert "### `08_encoder.py`" in methods and model in methods

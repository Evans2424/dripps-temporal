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

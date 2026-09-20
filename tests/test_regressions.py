"""Regression tests for bugs found in code review.

Each test pins a failure that shipped with a green suite, usually because a
definition was duplicated between the library and a script.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone, is_classifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dripps import evaluate, features, interpret, io, models, schema  # noqa: E402


@pytest.fixture(scope="module")
def df():
    return io.load()


# --- the auxiliary pattern was duplicated and one copy was double-escaped ----

def test_auxiliary_pattern_actually_matches():
    assert re.search(schema.APC_AUXILIARY["pt"], "x tendo y", re.I)
    assert re.search(schema.APC_AUXILIARY["en"], "x having y", re.I)


def test_span_recovery_counts(df):
    aux = io.has_apc_auxiliary(df)
    assert aux[df.is_portuguese].sum() == 793
    assert aux[~df.is_portuguese].sum() == 199
    assert (io.count_apc_auxiliary(df) > 1).sum() == 24


def test_audit_does_not_hardcode_span_counts():
    """The audit must compute these, not restate them as prose constants."""
    src = (ROOT / "experiments/01_audit.py").read_text(encoding="utf-8")
    assert "has_apc_auxiliary" in src
    assert "/793" not in src and "/200 " not in src


# --- **params made get_params() return {} and clone() drop every setting -----

def test_xgboost_params_survive_clone():
    m = models.b5_xgboost()
    assert m.get_params()["n_estimators"] == 400
    c = clone(m)
    assert (c.n_estimators, c.max_depth, c.random_state) == (400, 3, schema.SEED)


def test_custom_estimators_are_classifiers():
    assert is_classifier(models.b1_oner())
    assert is_classifier(models.b5_xgboost())


# --- derived columns were permuted without being recomputed -----------------

def test_recompute_derived_restores_products():
    X = pd.DataFrame({
        "mc_telic": [1, 0, 1], "sc_telic": [1, 1, 0],
        "mc_durative": [0, 1, 1], "sc_durative": [1, 1, 1],
        "both_telic": [9, 9, 9], "both_durative": [9, 9, 9],
    })
    features.recompute_derived(X)
    assert X["both_telic"].tolist() == [1, 0, 0]
    assert X["both_durative"].tolist() == [0, 1, 1]


def test_built_features_are_internally_consistent(df):
    X = features.build(df)
    assert (X["both_telic"] == X["mc_telic"] * X["sc_telic"]).all()
    assert (X["both_durative"] == X["mc_durative"] * X["sc_durative"]).all()


def test_permutation_skips_derived_only_blocks(df):
    """Permuting a derived block is a no-op once recomputed; ablation covers it."""
    sub = df[df.variety == "EP"]
    X, y, g = features.build(sub), features.target(sub), features.groups(sub)
    perm = interpret.block_permutation_importance(
        models.b3_logit(), X, y, g, features.CUE_BLOCKS, n_repeats=2
    )
    assert "interaction" not in set(perm["block"])
    abl = interpret.block_ablation_importance(
        models.b3_logit(), X, y, g, features.CUE_BLOCKS
    )
    assert "interaction" in set(abl["block"])


# --- macro-F1 averaged over a different label set per fold ------------------

def test_macro_f1_uses_a_fixed_label_set():
    y = ["Ant", "Ant", "Post"]
    labels = ["Ant", "Post", "Simul"]
    # Simul is absent from y; scoring must still average over all three
    a = evaluate.macro_f1(y, ["Ant", "Ant", "Post"], labels)
    b = evaluate.macro_f1(y, ["Ant", "Ant", "Post"], ["Ant", "Post"])
    assert a < b, "absent classes must count toward the macro average"


# --- CI came from CV repeats rather than from resampling the data -----------

def test_ci_reflects_sampling_not_seed_jitter(df):
    sub = io.analysis_frame(df)
    X, y, g = features.build(sub), features.target(sub), features.groups(sub)
    r = evaluate.score(models.b0_majority(), X, y, g, n_repeats=2)
    # a deterministic model has zero seed jitter but must still get a real interval
    assert r["sd_seed"] == pytest.approx(0.0, abs=1e-12)
    assert r["ci_hi"] - r["ci_lo"] > 0.005


# --- loader accepted a row with no sentence ---------------------------------

def test_truncated_row_is_rejected(tmp_path):
    lines = (ROOT / "data/raw/dripps_full.csv").read_text(encoding="utf-8").split("\n")
    fields = lines[1].split(";")
    lines[1] = ";".join(fields[:9] + [""])  # metadata + trailing empty, no Sentence
    bad = tmp_path / "truncated.csv"
    bad.write_text("\n".join(lines), encoding="utf-8")
    with pytest.raises(io.SchemaError):
        io.load(bad)


def test_unknown_discourse_relation_is_rejected(tmp_path):
    lines = (ROOT / "data/raw/dripps_full.csv").read_text(encoding="utf-8").split("\n")
    lines[1] = lines[1].replace(";asynchrony;", ";NOT-A-RELATION;")
    bad = tmp_path / "bad_dr.csv"
    bad.write_text("\n".join(lines), encoding="utf-8")
    with pytest.raises(io.SchemaError, match="NOT-A-RELATION"):
        io.load(bad)


# --- coefficients broke on two-class subsets (RQ2/RQ3 will hit these) -------

def test_logit_coefficients_handle_two_classes():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(60, 3)), columns=list("abc"))
    y = np.array(["Ant", "Post"] * 30)
    pipe = models.b3_logit().fit(X, y)
    coefs = interpret.logit_coefficients(pipe, X.columns, pipe[-1].classes_)
    assert coefs.shape == (3, 2)
    assert np.allclose(coefs["Ant"], -coefs["Post"])


# --- paths were resolved against the process cwd ----------------------------

def test_default_path_is_repo_anchored(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    assert io.DEFAULT_RAW.is_absolute() and io.DEFAULT_RAW.exists()
    assert len(io.load()) == 993

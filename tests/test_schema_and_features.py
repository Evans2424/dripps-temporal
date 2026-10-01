"""Guardrail tests: the mappings must be exhaustive and the leakage guard must bite."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dripps import features, io, schema  # noqa: E402
from dripps.leakage import LeakageError, assert_no_leakage  # noqa: E402


@pytest.fixture(scope="module")
def df():
    return io.load()


# --- mappings are exhaustive -------------------------------------------------

def test_tam_covers_every_attested_tense(df):
    unmapped = sorted(set(df["TMC"]) - set(schema.TAM_BUNDLE))
    assert not unmapped, f"TMC labels missing from TAM_BUNDLE: {unmapped}"


def test_aspect_covers_every_attested_class(df):
    for col in ("ATMC", "ATSC"):
        unmapped = sorted(set(df[col]) - set(schema.ASPECT_PRIMITIVES))
        assert not unmapped, f"{col} labels missing from ASPECT_PRIMITIVES: {unmapped}"


def test_aspect_primitives_are_distinct():
    """Each class must be uniquely recoverable from its three primitives."""
    assert len(set(schema.ASPECT_PRIMITIVES.values())) == len(schema.ASPECT_PRIMITIVES)


def test_moens_steedman_definitions():
    """Spot-check against the published event-nucleus taxonomy."""
    assert schema.ASPECT_PRIMITIVES["St"] == (False, True, False)    # state
    assert schema.ASPECT_PRIMITIVES["Culm"] == (True, False, True)   # culmination
    assert schema.ASPECT_PRIMITIVES["CP"] == (True, True, True)      # culminated process
    assert schema.ASPECT_PRIMITIVES["Pro"] == (True, True, False)    # process
    assert schema.ASPECT_PRIMITIVES["Pon"] == (True, False, False)   # point


def test_tam_bundle_fields_complete():
    for label, bundle in schema.TAM_BUNDLE.items():
        assert set(bundle) == set(schema.TAM_FIELDS), label
        assert bundle["tense"] in schema.TENSE_VALUES, label


def test_english_and_portuguese_tense_labels_are_disjoint_except_shared(df):
    """Motivates the TAM bundle: the raw label sets barely overlap."""
    pt = set(df.loc[df.is_portuguese, "TMC"])
    en = set(df.loc[~df.is_portuguese, "TMC"])
    assert pt & en == {"Pres", "Fut", "Cond", "GS"}, sorted(pt & en)


# --- data integrity ----------------------------------------------------------

def test_all_rows_parsed(df):
    assert len(df) == 1143
    assert df.variety.value_counts().to_dict() == {
        "EP": 250, "AP": 250, "MP": 250, "BE": 200, "BP": 193
    }


def test_second_batch_ids_are_distinct_and_tagged(df):
    """The Violeta batch carries a 'V' ID prefix, has no SR-SC, and is 50 per variety."""
    v = df[df.batch == "violeta"]
    assert len(v) == 150 and df.ID.is_unique
    assert v.ID.str.match(r"PT(EU|AO|MZ)V\d+$").all()
    assert not df.loc[df.batch == "original", "ID"].str.contains("V").any()
    assert v.variety.value_counts().to_dict() == {"EP": 50, "AP": 50, "MP": 50}
    assert (v["SR-SC"] == "").all() and (df.loc[df.batch == "original", "SR-SC"] != "").all()


def test_exotic_whitespace_is_normalized(df):
    """1133 NBSPs and one U+2028 must not reach tokenizers or span regexes."""
    assert df["Sentence"].str.contains("\u00a0").sum() == 557
    assert not df["sentence_norm"].str.contains("[\u00a0\u2028]").any()


def test_apc_auxiliary_is_recoverable(df):
    """The APC span is not annotated; it is recovered from the auxiliary."""
    pt = df[df.is_portuguese]["sentence_norm"]
    en = df[~df.is_portuguese]["sentence_norm"]
    assert pt.str.contains(r"\btendo\b", case=False).sum() == 943   # 100%
    assert en.str.contains(r"\bhaving\b", case=False).sum() == 199  # 199/200


def test_semicolon_sentences_recovered(df):
    """17 sentences contain a literal ';'; none may be dropped or truncated."""
    assert df["Sentence"].str.contains(";").sum() == 17
    assert (df["Sentence"].str.strip() == "").sum() == 0


def test_unknown_label_is_an_error(tmp_path):
    # split on "\n" only: the corpus contains a literal U+2028, which
    # str.splitlines() would treat as a line break and corrupt the file.
    raw = Path("data/raw/dripps_full.csv").read_text(encoding="utf-8").split("\n")
    raw[1] = raw[1].replace(";PP;", ";NOT-A-TENSE;")
    bad = tmp_path / "bad.csv"
    bad.write_text("\n".join(raw), encoding="utf-8")
    with pytest.raises(io.SchemaError, match="NOT-A-TENSE"):
        io.load(bad)


# --- leakage guard -----------------------------------------------------------

def test_guard_rejects_leaky_columns():
    for col in schema.LEAKY_COLUMNS:
        with pytest.raises(LeakageError):
            assert_no_leakage(["mc_telic", col])


def test_guard_rejects_one_hot_expansions():
    with pytest.raises(LeakageError):
        assert_no_leakage(["mc_telic", "DR_asynchrony", "SR-SC_before"])


def test_guard_allows_explicit_circular_baseline():
    assert_no_leakage(["DR"], allow_circular=True)


def test_built_features_never_leak(df):
    X = features.build(df)
    assert_no_leakage(X.columns)
    assert list(X.columns) == features.FEATURE_NAMES


# --- grouping ----------------------------------------------------------------

def test_multi_apc_sentences_share_a_group(df):
    """Duplicated sentences must never be split across CV folds."""
    per_group = df.groupby("sentence_group")["Sentence"].nunique()
    assert (per_group == 1).all()
    sizes = df.groupby("sentence_group").size()
    assert (sizes > 1).sum() == 26, "expected 26 multi-APC sentence groups"


# --- documented facts the paper depends on -----------------------------------

def test_leakage_is_definitional(df):
    """'before' is always anterior and 'after' always posterior -- by definition."""
    sr = df[df["SR-SC"].isin(["before", "after"])]
    assert set(sr.loc[sr["SR-SC"] == "before", "TR"]) == {"Ant"}
    assert set(sr.loc[sr["SR-SC"] == "after", "TR"]) == {"Post"}


def test_asynchrony_never_simultaneous(df):
    assert "Simul" not in set(df.loc[df["DR"] == "asynchrony", "TR"])


def test_english_is_near_categorically_anterior(df):
    be = df[df.variety == "BE"]
    assert (be["TR"] == "Ant").mean() == pytest.approx(0.95, abs=0.005)


def test_ep_perfeito_composto_forces_simultaneity(df):
    """EP's pretérito perfeito composto is iterative/durative, unlike EN present perfect."""
    ep = df[(df.variety == "EP") & (df["TMC"] == "PPC-Ind")]
    assert len(ep) == 6
    assert set(ep["TR"]) == {"Simul"}

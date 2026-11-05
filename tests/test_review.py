"""The annotators' corrections (data/raw/annotator_review.csv) are an opt-in layer over the export."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dripps import features, io  # noqa: E402


@pytest.fixture(scope="module")
def frames():
    return io.load(), io.load(reviewed=True)


def test_default_load_is_the_original_export(frames):
    orig, rev = frames
    assert "review" not in orig.columns and len(orig) == len(rev)


def test_corrections_change_exactly_the_reviewed_cells(frames):
    orig, rev = frames
    spec = pd.read_csv(io.REVIEW_RAW, dtype=str, keep_default_na=False)
    changed = {c: (orig[c] != rev[c]).sum() for c in ("TR", "ATMC", "ATSC")}
    assert changed == {c: int((spec[c] != "").sum()) for c in changed} == {"TR": 9, "ATMC": 2, "ATSC": 4}
    other = [c for c in orig.columns if c not in ("TR", "ATMC", "ATSC")]
    assert orig[other].equals(rev[other])  # sentence groups are unchanged; the folds are not (stratified on TR), so re-pair models
    assert rev.review.value_counts().to_dict() == {"": len(rev) - 30, "confirmed": 15, "changed": 11, "unsure": 4}


def test_unsure_and_confirmed_rows_keep_their_labels(frames):
    orig, rev = frames
    keep = rev.review.isin(["confirmed", "unsure"])
    assert (orig.loc[keep, ["TR", "ATMC", "ATSC"]] == rev.loc[keep, ["TR", "ATMC", "ATSC"]]).all().all()


def test_reviewed_frame_passes_validation_and_builds_features(frames):
    _, rev = frames
    X = features.build(io.analysis_frame(rev))
    assert X.shape[0] == len(io.analysis_frame(rev)) and not X.isna().any().any()


def test_a_stale_correction_fails_instead_of_overwriting():
    orig = io.load()
    spec = pd.read_csv(io.REVIEW_RAW, dtype=str, keep_default_na=False)
    for col, row in (("TR_was", spec.TR != ""), ("ATSC_was", spec.TR == ""), ("TR_was", spec.status == "confirmed")):
        bad = spec[row].copy()  # a changed cell, an unchanged column of another row, a confirmed row: all must be checked
        bad[col] = "Simul" if col == "TR_was" else "Culm"
        bad = bad[bad[col] != spec.loc[bad.index, col]]
        with pytest.raises(io.SchemaError, match="expects another"):
            io.apply_review(orig, bad)


def test_review_rejects_a_custom_path():
    with pytest.raises(ValueError, match="bundled"):
        io.load(io.ORIGINAL_RAW, reviewed=True)

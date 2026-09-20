"""Feature construction for the cue-hierarchy models.

Two linguistically-motivated recodings do the work:

* **Aspectual class -> Moens & Steedman (1988) primitives.** The 4-5 attested
  classes become three binaries (dynamic, durative, telic). This both shrinks
  the parameter count and states Silvano et al.'s (2021) hypothesis directly:
  anteriority/posteriority track telicity, simultaneity tracks durativity.
  Because that claim is about *both* clauses, the matching interactions are
  built explicitly rather than left for a model to discover.

* **Tense -> TAM bundle.** 31 tense labels, with disjoint Portuguese and English
  vocabularies, collapse into six orthogonal dimensions that are estimable at
  n~200/variety and comparable across languages.
"""

from __future__ import annotations

import pandas as pd

from . import schema
from .leakage import assert_no_leakage

#: Cue blocks, used for grouped permutation importance and ablation.
CUE_BLOCKS = {
    "position": ["pos_nonfinal"],
    "connector": ["has_connector"],
    "aspect_mc": ["mc_dynamic", "mc_durative", "mc_telic"],
    "aspect_sc": ["sc_dynamic", "sc_durative", "sc_telic"],
    "tense_mc": [
        "tense_present", "tense_future", "tense_nonfinite",
        "mc_perfective", "mc_progressive", "mc_perfect", "mc_irrealis",
    ],
    "interaction": ["both_telic", "both_durative"],
}

FEATURE_NAMES = [f for block in CUE_BLOCKS.values() for f in block]

#: Columns that are deterministic functions of other columns. Any procedure that
#: perturbs the inputs must recompute these, or it creates rows that cannot occur
#: (``both_telic=1`` with ``mc_telic=0``) and measures the model's response to
#: impossible input instead of the cue's contribution.
DERIVED = {
    "both_telic": ("mc_telic", "sc_telic"),
    "both_durative": ("mc_durative", "sc_durative"),
}

#: Blocks consisting only of derived columns. Permuting them is meaningless --
#: recomputation undoes it by construction -- so they are measured by refit
#: ablation only.
DERIVED_ONLY_BLOCKS = ("interaction",)


def recompute_derived(X):
    """Restore the derived columns from their sources, in place."""
    for target_col, (a, b) in DERIVED.items():
        if target_col in X.columns and a in X.columns and b in X.columns:
            X[target_col] = X[a].to_numpy() * X[b].to_numpy()
    return X


def _aspect(series: pd.Series, prefix: str) -> pd.DataFrame:
    prim = series.map(schema.ASPECT_PRIMITIVES)
    if prim.isna().any():
        bad = sorted(set(series[prim.isna()]))
        raise ValueError(f"Unmapped aspectual class(es): {bad}")
    return pd.DataFrame(
        {
            f"{prefix}_dynamic": prim.str[0],
            f"{prefix}_durative": prim.str[1],
            f"{prefix}_telic": prim.str[2],
        },
        index=series.index,
    ).astype(int)


def _tam(series: pd.Series) -> pd.DataFrame:
    bundle = series.map(schema.TAM_BUNDLE)
    if bundle.isna().any():
        bad = sorted(set(series[bundle.isna()]))
        raise ValueError(
            f"Unmapped tense label(s): {bad}. Add them to schema.TAM_BUNDLE -- "
            "do not let them fall through to a default."
        )
    tense = bundle.str["tense"]
    out = pd.DataFrame(index=series.index)
    # "past" is the reference level (the modal value in every variety)
    for value in ("present", "future", "nonfinite"):
        out[f"tense_{value}"] = (tense == value).astype(int)
    for field in ("perfective", "progressive", "perfect", "irrealis"):
        out[f"mc_{field}"] = bundle.str[field].astype(int)
    return out


def build(df: pd.DataFrame, *, allow_circular: bool = False) -> pd.DataFrame:
    """Build the model matrix. Raises if a definitionally circular column leaks in."""
    parts = [
        # Final is the reference level (85% of the corpus); the contrast of
        # interest is whether a non-final clause reads differently.
        pd.DataFrame(
            {"pos_nonfinal": (df["Position"] != "Final").astype(int)}, index=df.index
        ),
        pd.DataFrame(
            {"has_connector": (df["CNT"].str.strip() != "").astype(int)}, index=df.index
        ),
        _aspect(df["ATMC"], "mc"),
        _aspect(df["ATSC"], "sc"),
        _tam(df["TMC"]),
    ]
    X = pd.concat(parts, axis=1)
    # Silvano et al.'s "in both clauses" claim is an interaction claim.
    X["both_telic"] = X["mc_telic"] * X["sc_telic"]
    X["both_durative"] = X["mc_durative"] * X["sc_durative"]

    X = X[FEATURE_NAMES]
    assert_no_leakage(X.columns, allow_circular=allow_circular)
    return X


def target(df: pd.DataFrame) -> pd.Series:
    return df[schema.TARGET]


def groups(df: pd.DataFrame) -> pd.Series:
    """Grouping key for cross-validation: multi-APC sentences share a group."""
    return df["sentence_group"]

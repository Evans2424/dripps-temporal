"""Loading and validation for the DRIPPS export.

The exporter writes a ``;``-delimited file with a trailing empty field. Rather
than coercing unknown labels, ``load`` fails loudly: an unseen tense label means
the TAM mapping in :mod:`dripps.schema` is incomplete, which would silently
corrupt every downstream cue estimate.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from . import schema

#: Anchored to the repo, not the process cwd, so imports work from anywhere.
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW = ROOT / "data/raw/dripps_full.csv"
#: Second annotation batch, converted by experiments/00_ingest.py (IDs carry a "V").
VIOLETA_RAW = ROOT / "data/raw/dripps_violeta.csv"


class SchemaError(ValueError):
    """Raised when the export does not match the expected schema."""


def _read_unquoted(path: str | Path) -> pd.DataFrame:
    """Parse the ``;``-delimited export, which does not quote its fields.

    17 of the 993 sentences contain a literal ``;``, so a naive split yields
    12-18 fields instead of 11. ``Sentence`` is the last content field, so the
    leading metadata fields are split off positionally and the remainder is
    rejoined -- recovering the text rather than dropping the row.
    """
    n_meta = len(schema.RAW_COLUMNS) - 1  # every column except Sentence
    records = []
    with open(path, encoding="utf-8") as fh:
        header = next(fh).rstrip("\n").rstrip("\r").split(";")
        header = [h.strip() for h in header if h.strip()]
        if header != list(schema.RAW_COLUMNS):
            raise SchemaError(f"Unexpected header: {header}")
        for lineno, line in enumerate(fh, start=2):
            line = line.rstrip("\n").rstrip("\r")
            if not line.strip():
                continue
            fields = line.split(";")
            if len(fields) < n_meta + 2:
                raise SchemaError(
                    f"Line {lineno}: {len(fields)} fields, expected at least "
                    f"{n_meta + 2} ({n_meta} metadata + sentence + trailing empty)"
                )
            meta = fields[:n_meta]
            trailing = fields[-1]
            if trailing.strip():
                raise SchemaError(
                    f"Line {lineno}: expected empty trailing field, got {trailing!r}"
                )
            sentence = ";".join(fields[n_meta:-1])
            records.append(meta + [sentence])
    return pd.DataFrame(records, columns=list(schema.RAW_COLUMNS), dtype=str)


def _variety(series: pd.Series) -> pd.DataFrame:
    prefix = series.str.replace(r"\d+$", "", regex=True)
    unknown = sorted(set(prefix) - set(schema.VARIETY_BY_PREFIX))
    if unknown:
        raise SchemaError(f"Unknown ID prefix(es): {unknown}")
    mapped = prefix.map(schema.VARIETY_BY_PREFIX)
    return pd.DataFrame(
        {
            "variety": mapped.str[0],
            "language": mapped.str[1],
            "is_portuguese": mapped.str[2],
        },
        index=series.index,
    )


def load(path: str | Path | None = None, *, strict: bool = True) -> pd.DataFrame:
    """Load the export (both batches by default), validate every label, and
    attach variety metadata.

    Adds ``batch`` (``violeta`` for the second annotation batch, else
    ``original``), ``variety``, ``language``, ``is_portuguese``, and ``sentence_group``
    (a stable integer id shared by the duplicated rows of multi-APC sentences,
    used as the grouping key for every cross-validation split).
    """
    paths = [path] if path else [DEFAULT_RAW, VIOLETA_RAW]
    df = pd.concat([_read_unquoted(p) for p in paths], ignore_index=True)

    missing = [c for c in schema.RAW_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"Missing expected column(s): {missing}")

    for col in schema.RAW_COLUMNS:
        df[col] = df[col].str.strip()

    if strict:
        _validate(df)

    df = pd.concat([df, _variety(df[schema.ID])], axis=1)
    df["batch"] = df[schema.ID].str.match(r"PT[A-Z]{2}V\d+$").map({True: "violeta", False: "original"})
    # multi-APC sentences are duplicated across rows; group so CV never splits them
    df["sentence_norm"] = df[schema.SENTENCE].map(normalize_text)
    df["sentence_group"] = df.groupby("sentence_norm", sort=False).ngroup()
    return df


#: Characters the exporter emits inside sentence text: U+00A0 in 557 sentences
#: and U+2028 in one (``docs/DATA_AUDIT.md`` carries the exact counts); both
#: survive naive processing but change tokenization, so downstream text work
#: uses ``sentence_norm``.
_ODD_WHITESPACE = re.compile(r"[\u00a0\u2028\u2029\u000b\u000c\u0085]")


def normalize_text(text: str) -> str:
    """Collapse exotic whitespace so tokenizers and span regexes behave."""
    return re.sub(r"\s+", " ", _ODD_WHITESPACE.sub(" ", text)).strip()


def _validate(df: pd.DataFrame) -> None:
    checks = {
        schema.TARGET: schema.TR_LABELS,
        "Position": schema.POSITION_LABELS,
        "ATMC": schema.ASPECT_LABELS,
        "ATSC": schema.ASPECT_LABELS,
        "TMC": schema.TMC_LABELS,
        "DR": schema.DR_LABELS,
        "SR-SC": (*schema.SR_LABELS, ""),  # "" = not annotated (second batch)
    }
    problems = []
    for col, allowed in checks.items():
        unknown = sorted(set(df[col]) - set(allowed))
        if unknown:
            problems.append(f"  {col}: {unknown} (allowed: {list(allowed)[:6]}...)")
    conn = sorted({c for c in df["CNT"] if c} - set(schema.CONNECTOR_LABELS))
    if conn:
        problems.append(f"  CNT: {conn}")
    empty = df.index[df[schema.SENTENCE].str.strip() == ""].tolist()
    if empty:
        problems.append(f"  empty Sentence at row(s): {empty[:10]}")
    if df[schema.ID].duplicated().any():
        dupes = df.loc[df[schema.ID].duplicated(), schema.ID].tolist()
        problems.append(f"  duplicate IDs: {dupes[:10]}")
    if problems:
        raise SchemaError(
            "Export does not match schema -- update dripps.schema before proceeding:\n"
            + "\n".join(problems)
        )


def analysis_frame(df: pd.DataFrame, *, portuguese_only: bool = True) -> pd.DataFrame:
    """Restrict to the rows used for the RQ1/RQ2 cue-hierarchy analysis.

    British English is excluded by default: at 95% anterior its cue weights are
    unidentifiable, so it serves as a contrastive reference rather than a fifth
    variety (see docs/DATA_AUDIT.md).
    """
    return df[df["is_portuguese"]].copy() if portuguese_only else df.copy()


def has_apc_auxiliary(df: pd.DataFrame) -> pd.Series:
    """Whether each sentence contains the APC auxiliary for its language.

    The APC span is not annotated, so downstream span work keys off this token.
    """
    out = pd.Series(False, index=df.index)
    for lang, pattern in schema.APC_AUXILIARY.items():
        mask = df["language"] == lang
        out.loc[mask] = df.loc[mask, "sentence_norm"].str.contains(
            pattern, case=False, regex=True
        )
    return out


def count_apc_auxiliary(df: pd.DataFrame) -> pd.Series:
    """Number of APC auxiliaries per sentence; >1 means the span is ambiguous."""
    out = pd.Series(0, index=df.index)
    for lang, pattern in schema.APC_AUXILIARY.items():
        mask = df["language"] == lang
        out.loc[mask] = df.loc[mask, "sentence_norm"].str.count(f"(?i){pattern}")
    return out

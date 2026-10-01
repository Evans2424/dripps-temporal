"""Read the tables the experiment scripts write, naming the step that is missing.

Shared by the static viewer (``experiments/06_viewer.py``) and the interactive
app (``app/``), so both fail the same way and a new table is registered once.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "results/tables"

#: table -> the make target that writes it
PRODUCED_BY = {
    "baselines.csv": "make baselines",
    "cue_importance.csv": "make hierarchy",
    "logit_coefficients.csv": "make hierarchy",
    "marginal_association.csv": "make hierarchy",
    "aspect_adjusted.csv": "make hierarchy",
    "oner_rule.json": "make hierarchy",
    "tree_rules.txt": "make hierarchy",
    "variety_cue_importance.csv": "make varieties",
    "variety_cue_ranks.csv": "make varieties",
    "variety_performance.csv": "make varieties",
    "shap_summary.csv": "make explain",
    "shap_blocks.csv": "make explain",
    "shap_forest.csv": "make explain",
    "shap_xgboost.csv": "make explain",
}


class MissingTable(FileNotFoundError):
    """A results table is absent; the message names the target that writes it."""


def path(name: str) -> Path:
    p = TABLES / name
    if not p.exists():
        raise MissingTable(
            f"missing {p.relative_to(ROOT)} -- run `{PRODUCED_BY[name]}` first"
        )
    return p


def csv(name: str, **kw) -> pd.DataFrame:
    return pd.read_csv(path(name), **kw)


def text(name: str) -> str:
    return path(name).read_text(encoding="utf-8")


def json_table(name: str):
    return json.loads(text(name))

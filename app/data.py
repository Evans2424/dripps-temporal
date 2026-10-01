"""Cached access to the corpus, the results tables and the fitted models.

Everything expensive is cached: the tables with ``st.cache_data`` and fitted
models with ``st.cache_resource``. Out-of-fold predictions use the same
sentence-grouped folds as ``evaluate.cv_predict`` (one repeat, to stay quick),
so what the app shows per sentence is what the reported scores are built from.
"""

from __future__ import annotations

import re
import sys
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from dripps import evaluate, features, interpret, io, models, results, schema  # noqa: E402

READINGS = list(schema.TR_LABELS)
COLORS = {"Ant": "#8F4E15", "Post": "#4C4B9E", "Simul": "#0E7C7B"}
ASPECT_NAMES = {"St": "State", "Pro": "Process", "Culm": "Culmination",
                "CP": "Culminated process", "Pon": "Point"}
BLOCK_NAMES = {
    "aspect_mc": "main-clause aspect", "tense_mc": "main-clause tense",
    "position": "clause position", "connector": "connector",
    "aspect_sc": "participial-clause aspect", "interaction": "both-clause interactions",
}
CUE_COLUMNS = {"ATMC": "main-clause aspect", "ATSC": "participial aspect",
               "TMC": "main-clause tense", "Position": "position", "CNT": "connector"}


def table(name: str, **kw) -> pd.DataFrame | None:
    """A results table, or None with a visible hint if its step has not run."""
    try:
        return _table(name, tuple(sorted(kw.items())))
    except results.MissingTable as e:
        st.warning(str(e))
        return None


@st.cache_data(show_spinner=False)
def _table(name: str, kw: tuple) -> pd.DataFrame:
    return results.csv(name, **dict(kw))


@st.cache_data(show_spinner=False)
def corpus() -> pd.DataFrame:
    """All 993 rows, with readable cue columns added for filtering."""
    df = io.load().copy()
    df["connector"] = np.where(df["CNT"].str.strip() != "", df["CNT"].str.strip(), "none")
    df["position3"] = df["Position"]
    df["main aspect"] = df["ATMC"].map(ASPECT_NAMES)
    df["participial aspect"] = df["ATSC"].map(ASPECT_NAMES)
    df["n_tendo"] = io.count_apc_auxiliary(df)
    return df


@st.cache_data(show_spinner=False)
def matrices():
    df = io.analysis_frame(io.load())
    X, y, g = features.build(df), features.target(df), features.groups(df)
    return df, X, y, g


@st.cache_resource(show_spinner="Fitting B3 on the Portuguese set…")
def fitted_logit():
    _, X, y, _ = matrices()
    return models.b3_logit().fit(X, y)


@st.cache_resource(show_spinner="Fitting the depth-3 tree…")
def fitted_tree():
    _, X, y, _ = matrices()
    return models.b4_tree(max_depth=3).fit(X, y)


#: Not on the ladder: B3 plus one indicator per attested (main, participial)
#: aspect-class pair, so each clause combination gets its own weights. It tests
#: whether the two derived interactions miss a combination the errors need.
PAIR_MODEL = "B3 + clause pairs"
#: The models compared on the error-analysis page, B3 first.
ERROR_MODELS = ["B3 logit", PAIR_MODEL, "B4 tree(d=4)", "B5 forest", "B5 xgboost"]


def pair_matrix(df: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
    """X plus a one-hot of the main x participial aspect-class pair."""
    pair = pd.get_dummies(df["ATMC"].astype(str) + "×" + df["ATSC"].astype(str), prefix="pair", dtype=int)
    return pd.concat([X, pair.set_index(X.index)], axis=1)


@st.cache_data(show_spinner="Computing out-of-fold predictions…")
def oof(label: str) -> pd.DataFrame:
    """One repeat of sentence-grouped 5-fold predictions, with probabilities.

    Every label uses the same folds, so two models' predictions for a sentence
    come from models trained on exactly the same other sentences.
    """
    df, X, y, g = matrices()
    if label == PAIR_MODEL:
        X, make = pair_matrix(df, X), models.b3_logit
    else:
        make = models.LADDER[label]
    Xv, yv, gv = X.to_numpy(), y.to_numpy(), g.to_numpy()
    cv = StratifiedGroupKFold(n_splits=evaluate.N_SPLITS, shuffle=True, random_state=schema.SEED)
    pred = np.empty(len(yv), dtype=object)
    proba = np.full((len(yv), len(READINGS)), np.nan)
    for train, test in cv.split(Xv, yv, gv):
        est = clone(make()).fit(Xv[train], yv[train])
        pred[test] = est.predict(Xv[test])
        if hasattr(est, "predict_proba"):
            cols = list(est.classes_)
            p = est.predict_proba(Xv[test])
            for j, r in enumerate(READINGS):
                if r in cols:
                    proba[test, j] = p[:, cols.index(r)]
    out = pd.DataFrame(proba, columns=[f"p_{r}" for r in READINGS])
    out.insert(0, "pred", pred)
    out.insert(0, "gold", yv)
    out.insert(0, "ID", df[schema.ID].to_numpy())
    return out


@st.cache_data(show_spinner="Comparing the models sentence by sentence…")
def predictions() -> pd.DataFrame:
    """One row per Portuguese clause: annotation, B3's prediction and confidence,
    and whether each comparison model got it right (same folds throughout)."""
    b3 = oof("B3 logit")
    p = b3[[f"p_{r}" for r in READINGS]].to_numpy()
    top2 = np.sort(p, axis=1)[:, -2:]
    out = b3[["ID", "gold", "pred"]].assign(
        confidence=top2[:, 1],
        margin=top2[:, 1] - top2[:, 0],
        p_gold=p[np.arange(len(p)), [READINGS.index(g) for g in b3["gold"]]],
    )
    for label in ERROR_MODELS:
        out[f"ok · {label}"] = (oof(label)["pred"] == oof(label)["gold"]).to_numpy()
    out["models wrong"] = (~out[[f"ok · {m}" for m in ERROR_MODELS]]).sum(axis=1)
    out["participle"] = [participle(s) for s in corpus().set_index("ID").loc[out["ID"], "Sentence"]]
    return out


def participle(sentence: str) -> str:
    """The first participle after the auxiliary, lower-cased; '' if none is found."""
    for m in _AUX.finditer(sentence):
        for w in sentence[m.end():].split()[:4]:
            word = w.rstrip(",.;:)").lower()
            if PARTICIPLE.match(word) and word not in ("sido", "been"):
                return word
    return ""


def features_for(position: str, connector: bool, atmc: str, atsc: str, tmc: str) -> pd.DataFrame:
    """One model row from annotation values, through the same code path as the corpus."""
    row = pd.DataFrame([{"Position": position, "CNT": "mesmo" if connector else "",
                         "ATMC": atmc, "ATSC": atsc, "TMC": tmc}])
    return features.build(row)


def logit_contributions(x_row: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, list]:
    """Per-feature push (standardised value x weight) on each reading's score."""
    pipe = fitted_logit()
    scaler, clf = pipe[0], pipe[-1]
    z = scaler.transform(x_row)[0]
    contrib = pd.DataFrame(z[:, None] * clf.coef_.T, index=x_row.columns, columns=clf.classes_)
    scores = clf.intercept_ + contrib.sum().to_numpy()
    return contrib, scores, list(clf.classes_)


@st.cache_data(show_spinner=False)
def shap_long(slug: str) -> pd.DataFrame:
    return results.csv(f"shap_{slug}.csv")


@st.cache_data(show_spinner=False)
def aspect_adjusted() -> pd.DataFrame:
    return results.csv("aspect_adjusted.csv")


def tree_path(row_index: int) -> list[str]:
    """The questions the tree asks one sentence, in order."""
    _, X, _, _ = matrices()
    tree = fitted_tree()
    t = tree.tree_
    node_ids = tree.decision_path(X.iloc[[row_index]]).indices
    steps = []
    for n in node_ids:
        if t.children_left[n] == -1:
            counts = t.value[n][0] * t.n_node_samples[n]
            steps.append(f"leaf: predicts **{tree.classes_[int(np.argmax(counts))]}** "
                         f"({t.n_node_samples[n]} sentences)")
        else:
            f = X.columns[t.feature[n]]
            val = X.iloc[row_index, t.feature[n]]
            steps.append(f"`{f}` = {int(val)}")
    return steps


#: A Portuguese past participle: regular -ado/-ido forms plus the common irregulars.
#: A pattern, not a lexicon -- good enough to locate the clause, not to count verbs.
PARTICIPLE = re.compile(
    r"^(sido|ido|\w+(?:ad|id|íd)[oa]s?|feit[oa]s?|dit[oa]s?|vist[oa]s?|post[oa]s?|propost[oa]s?|"
    r"abert[oa]s?|escrit[oa]s?|cobert[oa]s?|descobert[oa]s?|mort[oa]s?|aceit[oa]s?|aceites?|"
    r"entregues?|ganh[oa]s?|pag[oa]s?|gast[oa]s?|eleit[oa]s?|pres[oa]s?|expuls[oa]s?|"
    r"suspens[oa]s?|impress[oa]s?|salv[oa]s?|solt[oa]s?|vindo|tido|\w+(?:ed|en|wn|ne))[,.;:)]?$", re.I)
_AUX = re.compile(r"(?i)\b(tendo|having)\b")


def highlight(sentence: str) -> str:
    """HTML with each auxiliary and its participle marked (up to 4 words later)."""
    spans = []
    for m in _AUX.finditer(sentence):
        spans.append((m.start(), m.end()))
        for w in list(re.finditer(r"\S+", sentence[m.end():]))[:4]:
            if PARTICIPLE.match(w.group(0)) and w.group(0).lower() not in ("sido", "been"):
                end = m.end() + w.end() - (1 if w.group(0)[-1] in ",.;:)" else 0)
                spans.append((m.end() + w.start(), end))
                break
    out, last = [], 0
    for a, b in sorted(spans):
        out.append(escape(sentence[last:a]))
        out.append(f"<mark>{escape(sentence[a:b])}</mark>")
        last = b
    out.append(escape(sentence[last:]))
    return "".join(out)


def model_labels() -> list[str]:
    return list(models.LADDER)


def blocks() -> dict:
    return features.CUE_BLOCKS


def tree_structure() -> dict:
    df, X, _, _ = matrices()
    tree = fitted_tree()
    return interpret.tree_structure(tree, X.columns, tree.classes_, X=X,
                                    ids=list(df[schema.ID]))

"""Cue-hierarchy extraction.

Importance is measured two ways, because neither alone is sound here.

**Refit ablation** (primary) drops a cue block and refits, answering "does the
model still work without this cue?". It is the only valid measure for the
interaction block, whose columns are deterministic products of the aspect
columns -- permuting those in place either fabricates impossible rows or, once
recomputed, is a no-op by construction.

**Block permutation** (secondary) keeps the fitted model and shuffles a block in
the held-out fold, answering "how much does the fitted model rely on this cue?".
Derived columns are recomputed after each shuffle. Whole blocks are permuted
together so correlated columns cannot mask one another.

Impurity importance is never used: gini rewards high cardinality, and the tense
block expands to seven columns against position's one.

Both measures pool predictions out-of-fold and take their interval from
resampling sentence groups, as ``evaluate.bootstrap_group_ci`` does, so ``ci_lo``
means the same thing here as it does there. Across-fold spread is reported
separately as ``sd_fold``; it is jitter of the split, not of the data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold

from .evaluate import macro_f1
from .features import DERIVED_ONLY_BLOCKS, recompute_derived
from .schema import ASPECT_PRIMITIVES, SEED


def _folds(X, y, groups, n_splits, seed):
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(cv.split(X, y, groups))


def _group_bootstrap_ci(y, groups, base, variants, *, n_boot, alpha=0.05, seed=SEED):
    """Percentile CI for a *drop* in macro-F1, resampling sentence groups.

    ``base`` and every vector in ``variants`` are out-of-fold predictions over
    the same rows, so a draw scores both on the identical resampled sentences
    and the pairing survives the resample. Where a block has several variants
    (the permutation repeats) a draw picks one, folding that jitter into the
    interval the way ``evaluate.bootstrap_group_ci`` folds in the fold seed.

    Quantiles of the five per-fold drops would instead describe how the estimate
    moves across fold assignments of the same sentences -- with ``n_splits=5``
    little more than their min and max. See the note in ``evaluate``.
    """
    labels = sorted(pd.unique(y))
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    rows_by_group = {g: np.flatnonzero(groups == g) for g in uniq}

    draws = np.empty(n_boot)
    for b in range(n_boot):
        picked = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([rows_by_group[g] for g in picked])
        var = variants[rng.integers(len(variants))]
        draws[b] = macro_f1(y[idx], base[idx], labels) - macro_f1(y[idx], var[idx], labels)
    return float(np.quantile(draws, alpha / 2)), float(np.quantile(draws, 1 - alpha / 2))


def _row(name, y, groups, base, variants, per_fold, labels, n_boot):
    full = macro_f1(y, base, labels)
    lo, hi = (
        _group_bootstrap_ci(y, groups, base, variants, n_boot=n_boot)
        if n_boot else (float("nan"), float("nan"))
    )
    return {
        "block": name,
        "importance": float(np.mean([full - macro_f1(y, v, labels) for v in variants])),
        "sd_fold": float(np.std(per_fold, ddof=1)),
        "ci_lo": lo,
        "ci_hi": hi,
    }


def _table(rows: list[dict]) -> pd.DataFrame:
    return (
        pd.DataFrame(rows).sort_values("importance", ascending=False).reset_index(drop=True)
    )


def block_ablation_importance(
    model, X: pd.DataFrame, y, groups, blocks: dict[str, list[str]],
    *, n_splits: int = 5, n_boot: int = 2000, seed: int = SEED,
) -> pd.DataFrame:
    """Drop in out-of-fold macro-F1 when a cue block is removed and the model refit.

    Correlated cues can compensate for a dropped block, so this is a test of
    *necessity*: a block scoring near zero is one the model can do without.

    The point estimate is the drop over pooled out-of-fold predictions and the
    interval comes from resampling sentence groups; ``sd_fold`` reports the
    across-fold jitter separately. Pass ``n_boot=0`` to skip the bootstrap when
    only the ranking is wanted.
    """
    y = np.asarray(y)
    groups = np.asarray(groups)
    labels = sorted(pd.unique(y))

    base = np.empty(len(y), dtype=object)
    reduced = {name: np.empty(len(y), dtype=object) for name in blocks}
    per_fold: dict[str, list[float]] = {name: [] for name in blocks}

    for train, test in _folds(X, y, groups, n_splits, seed):
        full = clone(model).fit(X.iloc[train], y[train])
        base[test] = full.predict(X.iloc[test])
        fold_base = macro_f1(y[test], base[test], labels)
        for name, cols in blocks.items():
            keep = [c for c in X.columns if c not in cols]
            # dropping an aspect block leaves its products dangling
            keep = [c for c in keep if c not in _orphaned_derived(cols, keep)]
            est = clone(model).fit(X.iloc[train][keep], y[train])
            reduced[name][test] = est.predict(X.iloc[test][keep])
            per_fold[name].append(fold_base - macro_f1(y[test], reduced[name][test], labels))

    return _table([
        _row(name, y, groups, base, [reduced[name]], per_fold[name], labels, n_boot)
        for name in blocks
    ])


def _orphaned_derived(removed: list[str], keep: list[str]) -> set[str]:
    from .features import DERIVED

    return {
        d for d, sources in DERIVED.items()
        if d in keep and any(src in removed for src in sources)
    }


def block_permutation_importance(
    model, X: pd.DataFrame, y, groups, blocks: dict[str, list[str]],
    *, n_repeats: int = 20, n_splits: int = 5, n_boot: int = 2000, seed: int = SEED,
) -> pd.DataFrame:
    """Drop in out-of-fold macro-F1 when a cue block is permuted in the test fold."""
    y = np.asarray(y)
    groups = np.asarray(groups)
    labels = sorted(pd.unique(y))
    rng = np.random.default_rng(seed)

    # recomputation makes permutation a no-op for derived blocks; use ablation
    live = [name for name in blocks if name not in DERIVED_ONLY_BLOCKS]
    base = np.empty(len(y), dtype=object)
    permuted = {n: [np.empty(len(y), dtype=object) for _ in range(n_repeats)] for n in live}
    per_fold: dict[str, list[float]] = {name: [] for name in live}

    for train, test in _folds(X, y, groups, n_splits, seed):
        est = clone(model).fit(X.iloc[train], y[train])
        X_test = X.iloc[test]
        base[test] = est.predict(X_test)
        fold_base = macro_f1(y[test], base[test], labels)
        for name in live:
            cols = blocks[name]
            Xp = X_test.copy()
            drops = []
            for r in range(n_repeats):
                order = rng.permutation(len(Xp))
                Xp[cols] = X_test[cols].to_numpy()[order]
                recompute_derived(Xp)
                permuted[name][r][test] = est.predict(Xp)
                drops.append(fold_base - macro_f1(y[test], permuted[name][r][test], labels))
            per_fold[name].append(float(np.mean(drops)))

    return _table([
        _row(name, y, groups, base, permuted[name], per_fold[name], labels, n_boot)
        for name in live
    ])


def _set_class(X, prefix, cls):
    Xc = X.copy()
    for name, value in zip(("dynamic", "durative", "telic"), ASPECT_PRIMITIVES[cls]):
        Xc[f"{prefix}_{name}"] = int(value)
    return recompute_derived(Xc)


def adjusted_class_probabilities(
    model, X, y, groups, *, prefix="mc", classes=("Culm", "Pro", "CP", "St"),
    n_boot: int = 500, alpha: float = 0.05, seed: int = SEED,
) -> pd.DataFrame:
    """Predicted reading distribution per aspectual class, other cues as observed.

    Every row is assigned the class in turn and the fitted model's probabilities
    are averaged over the corpus (g-computation), so tense, position and the
    other clause keep their real distribution and only aspect changes. Unlike a
    crosstab this is not confounded by which tenses each class happens to occur
    with. Intervals refit the model on sentence-group bootstrap resamples.
    """
    y = np.asarray(y)
    groups = np.asarray(groups)

    def estimate(Xf, yf):
        fitted = clone(model).fit(Xf, yf)
        return {c: fitted.predict_proba(_set_class(X, prefix, c)).mean(axis=0)
                for c in classes}, list(fitted.classes_)

    point, labels = estimate(X, y)
    uniq = np.unique(groups)
    rows_by_group = {g: np.flatnonzero(groups == g) for g in uniq}
    rng = np.random.default_rng(seed)
    draws = {c: [] for c in classes}
    for _ in range(n_boot):
        idx = np.concatenate([rows_by_group[g] for g in rng.choice(uniq, len(uniq))])
        est, lab = estimate(X.iloc[idx], y[idx])
        if lab != labels:  # a resample missing a reading cannot be compared
            continue
        for c in classes:
            draws[c].append(est[c])

    out = []
    for c in classes:
        d = np.array(draws[c])
        for j, reading in enumerate(labels):
            out.append({
                "clause": prefix, "aspect_class": c, "reading": reading,
                "prob": float(point[c][j]),
                "ci_lo": float(np.quantile(d[:, j], alpha / 2)),
                "ci_hi": float(np.quantile(d[:, j], 1 - alpha / 2)),
            })
    return pd.DataFrame(out)


def logit_coefficients(fitted_pipeline, feature_names, classes) -> pd.DataFrame:
    """Standardised coefficients, one column per outcome class.

    sklearn returns a single coefficient row for a two-class fit; it is expanded
    to the symmetric two-column form so per-variety and transfer analyses, which
    can hit two-class subsets, behave like the three-class case.
    """
    clf = fitted_pipeline[-1] if hasattr(fitted_pipeline, "__getitem__") else fitted_pipeline
    coef = np.asarray(clf.coef_)
    classes = list(classes)
    if coef.shape[0] == 1 and len(classes) == 2:
        coef = np.vstack([-coef[0], coef[0]])
    return pd.DataFrame(coef.T, index=list(feature_names), columns=classes)


def tree_rules(tree, feature_names, class_names, *, max_depth=None) -> str:
    from sklearn.tree import export_text

    return export_text(
        tree, feature_names=list(feature_names), max_depth=max_depth or 10,
        class_names=list(class_names), decimals=2,
    )


def tree_structure(tree, feature_names, class_names, *, X=None, ids=None) -> dict:
    """Node-by-node form of the same tree ``tree_rules`` renders as text.

    The text export is what the paper quotes; this carries what the text drops --
    sample counts, the class distribution at every node and, given ``X`` and
    ``ids``, the rows landing in each leaf -- so a reader can see which sentences
    a rule actually covers.
    """
    t = tree.tree_
    feature_names, class_names = list(feature_names), list(class_names)

    rows_by_leaf: dict[int, list] = {}
    if X is not None and ids is not None:
        for node, rid in zip(tree.apply(X), list(ids)):
            rows_by_leaf.setdefault(int(node), []).append(rid)

    nodes = []
    for i in range(t.node_count):
        leaf = t.children_left[i] == -1
        # tree_.value holds class *proportions*, not counts, in current sklearn
        counts = (t.value[i][0] * t.n_node_samples[i]).round().astype(int).tolist()
        nodes.append({
            "id": i,
            "is_leaf": bool(leaf),
            "feature": None if leaf else feature_names[t.feature[i]],
            "threshold": None if leaf else float(t.threshold[i]),
            "n_samples": int(t.n_node_samples[i]),
            "counts": counts,
            "predicted": class_names[int(np.argmax(counts))],
            "impurity": float(t.impurity[i]),
            "left": None if leaf else int(t.children_left[i]),
            "right": None if leaf else int(t.children_right[i]),
            "rows": rows_by_leaf.get(i, []),
        })
    return {"classes": class_names, "nodes": nodes}

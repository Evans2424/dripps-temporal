"""The baseline ladder (B0-B5).

Ordered so that each rung answers "does the extra machinery earn its place?".
B3 is the paper's primary inferential model; B4/B5 exist to find cue
*combinations* an additive model cannot express, and to check whether any
nonlinearity is real.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .schema import SEED


class OneR(ClassifierMixin, BaseEstimator):
    """Predict the majority class of the single most informative feature.

    The honest floor for a "cue hierarchy" claim: if the full model cannot beat
    the best individual cue, there is no hierarchy worth reporting.
    """

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        labels, counts = np.unique(y, return_counts=True)
        self.classes_ = labels
        fallback = labels[counts.argmax()]  # invariant across features
        best = (-1.0, 0, {}, fallback)
        for j in range(X.shape[1]):
            rules, correct = {}, 0
            for value in np.unique(X[:, j]):
                mask = X[:, j] == value
                labels, counts = np.unique(y[mask], return_counts=True)
                rules[value] = labels[counts.argmax()]
                correct += counts.max()
            acc = correct / len(y)
            if acc > best[0]:
                best = (acc, j, rules, fallback)
        self.train_accuracy_, self.feature_, self.rules_, self.default_ = best
        return self

    def predict(self, X):
        X = np.asarray(X)
        return np.array([self.rules_.get(v, self.default_) for v in X[:, self.feature_]])


def b0_majority():
    return DummyClassifier(strategy="most_frequent")


def b1_oner():
    return OneR()


def b3_logit(C: float = 1.0):
    """L2-penalised multinomial logistic regression -- primary model for RQ1."""
    return Pipeline([
        ("scale", StandardScaler()),
        # L2 is the default; sklearn>=1.8 deprecates the explicit penalty kwarg
        ("clf", LogisticRegression(C=C, max_iter=5000, random_state=SEED)),
    ])


def b4_tree(max_depth: int = 4, min_samples_leaf: int = 20):
    """Shallow enough that the splits can be read and reported verbatim."""
    return DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=SEED,
    )


def b5_forest(n_estimators: int = 1000):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=SEED,
    )


class _XGBWrapper(ClassifierMixin, BaseEstimator):
    """XGBoost with string labels, so every rung of the ladder shares an interface.

    Parameters are named explicitly rather than collected with ``**params``:
    sklearn's ``_get_param_names`` skips VAR_KEYWORD arguments, so a ``**params``
    signature makes ``get_params()`` return ``{}`` and ``clone()`` silently drop
    every hyperparameter -- including ``random_state``.
    """

    def __init__(self, n_estimators=400, max_depth=3, learning_rate=0.05,
                 subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                 random_state=SEED, n_jobs=-1, tree_method="hist"):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.tree_method = tree_method

    def fit(self, X, y):
        from xgboost import XGBClassifier

        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self._index = {c: i for i, c in enumerate(self.classes_)}
        self._model = XGBClassifier(**self.get_params())
        self._model.fit(X, np.array([self._index[v] for v in y]))
        return self

    def predict(self, X):
        return self.classes_[self._model.predict(X)]

    def predict_proba(self, X):
        return self._model.predict_proba(X)

    @property
    def booster_(self):
        """The fitted XGBClassifier, for tools that need the booster itself.

        Deliberately not named ``estimator_``: sklearn's own ensembles use that
        for the *unfitted* template estimator, so a tool unwrapping by that name
        would silently get a bare DecisionTreeClassifier from a RandomForest.

        Its classes are the integer codes in ``classes_`` order, so anything
        reading per-class output maps position *i* back through ``classes_[i]``.
        """
        return self._model


def b5_xgboost(**kw):
    return _XGBWrapper(**kw)


LADDER = {
    "B0 majority": b0_majority,
    "B1 one-rule": b1_oner,
    "B3 logit": b3_logit,
    "B4 tree(d=4)": b4_tree,
    "B5 forest": b5_forest,
    "B5 xgboost": b5_xgboost,
}

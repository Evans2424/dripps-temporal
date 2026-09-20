"""Guards against the definitional leakage in the DRIPPS annotation.

``SR-SC`` and ``DR`` are not merely correlated with ``TR`` -- they are partly
defined by it. In the released corpus ``SR-SC == "before"`` is anterior in 100%
of cases and ``"after"`` is posterior in 100% of cases (Cramer's V = 0.77);
``DR == "asynchrony"`` never co-occurs with ``Simul`` (V = 0.56).

A model using them would score near ceiling while explaining nothing. They are
reported once as an explicit "circular upper bound" (``circular_columns``) and
are otherwise excluded by ``assert_no_leakage``, which every model path calls.
"""

from __future__ import annotations

from .schema import LEAKY_COLUMNS


class LeakageError(ValueError):
    """Raised when a definitionally circular column reaches a TR model."""


def assert_no_leakage(columns, *, allow_circular: bool = False) -> None:
    """Raise if any leaky column appears in ``columns``.

    Parameters
    ----------
    columns:
        Column names about to be used as features.
    allow_circular:
        Only ``True`` for the deliberate circular-baseline experiment, which
        exists to demonstrate the leakage rather than to hide it.
    """
    if allow_circular:
        return
    names = {str(c) for c in columns}
    # one-hot expansion turns "DR" into "DR_cause" etc., so match on prefix too
    found = sorted(
        n for n in names
        if n in LEAKY_COLUMNS or any(n.startswith(f"{c}_") for c in LEAKY_COLUMNS)
    )
    if found:
        raise LeakageError(
            f"Definitionally circular column(s) in TR feature matrix: {found}. "
            f"{LEAKY_COLUMNS} encode temporal order by definition. "
            "Pass allow_circular=True only for the circular-baseline experiment."
        )


def circular_columns() -> tuple[str, ...]:
    """The leaky columns, for the deliberate upper-bound baseline."""
    return LEAKY_COLUMNS

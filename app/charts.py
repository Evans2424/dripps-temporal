"""Altair builders shared by the pages, so every chart uses one reading palette."""

from __future__ import annotations

import altair as alt
import pandas as pd

from data import COLORS, READINGS

READING_SCALE = alt.Scale(domain=READINGS, range=[COLORS[r] for r in READINGS])


def reading_color(title: str = "reading") -> alt.Color:
    return alt.Color("reading:N", scale=READING_SCALE, title=title,
                     sort=READINGS, legend=alt.Legend(orient="top"))


def stacked_share(counts: pd.DataFrame, category: str, *, normalize: bool = True,
                  sort=None, height: int | None = None) -> alt.Chart:
    """Horizontal 100% (or raw) stacked bars of readings per category.

    ``counts`` is long form with columns [category, reading, n].
    """
    total = counts.groupby(category)["n"].transform("sum")
    data = counts.assign(share=counts["n"] / total, total=total)
    x = alt.X("n:Q", stack="normalize" if normalize else True,
              title="share of clauses" if normalize else "clauses",
              axis=alt.Axis(format="%" if normalize else "d"))
    rows = data[category].nunique()
    return (
        alt.Chart(data)
        .mark_bar()
        .encode(
            y=alt.Y(f"{category}:N", sort=sort, title=None),
            x=x,
            color=reading_color(),
            order=alt.Order("reading_order:Q"),
            tooltip=[alt.Tooltip(f"{category}:N"), "reading:N", alt.Tooltip("n:Q", title="clauses"),
                     alt.Tooltip("share:Q", format=".0%"), alt.Tooltip("total:Q", title="row total")],
        )
        .transform_calculate(reading_order=f"indexof({READINGS!r}, datum.reading)")
        .properties(height=height or max(90, 34 * rows))
    )


def counts_by(df: pd.DataFrame, category: str) -> pd.DataFrame:
    """Long-form reading counts per value of ``category``, zeros included."""
    ct = pd.crosstab(df[category], df["TR"]).reindex(columns=READINGS, fill_value=0)
    return ct.reset_index().melt(id_vars=category, var_name="reading", value_name="n")


def interval_dots(data: pd.DataFrame, *, y: str, x: str, lo: str, hi: str,
                  color: str | None = None, sort=None, x_title: str = "") -> alt.LayerChart:
    """Point estimates with 95% interval whiskers and a zero rule."""
    enc_color = alt.Color(f"{color}:N", legend=alt.Legend(orient="top")) if color else alt.value("#0E7C7B")
    base = alt.Chart(data).encode(y=alt.Y(f"{y}:N", sort=sort, title=None))
    whisk = base.mark_rule(strokeWidth=2).encode(
        x=alt.X(f"{lo}:Q", title=x_title), x2=f"{hi}:Q", color=enc_color)
    dots = base.mark_circle(size=110, opacity=1).encode(
        x=f"{x}:Q", color=enc_color,
        tooltip=[f"{y}:N", alt.Tooltip(f"{x}:Q", format=".3f"),
                 alt.Tooltip(f"{lo}:Q", format=".3f", title="95% lo"),
                 alt.Tooltip(f"{hi}:Q", format=".3f", title="95% hi")])
    zero = alt.Chart(pd.DataFrame({"z": [0]})).mark_rule(strokeDash=[4, 4], color="#79888F").encode(x="z:Q")
    return (zero + whisk + dots)

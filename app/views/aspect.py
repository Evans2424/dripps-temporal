import altair as alt
import pandas as pd
import streamlit as st

from charts import reading_color
from data import ASPECT_NAMES, READINGS, aspect_adjusted, corpus, table

ORDER = ["Culmination", "Process", "Culminated process", "State"]


def _raw(pt: pd.DataFrame, col: str) -> pd.DataFrame:
    ct = pd.crosstab(pt[col], pt["TR"], normalize="index").reindex(columns=READINGS, fill_value=0)
    n = pt[col].value_counts()
    out = ct.reset_index().melt(id_vars=col, var_name="reading", value_name="prob")
    out["class"] = out[col].map(ASPECT_NAMES)
    out["n"] = out[col].map(n)
    return out[out[col] != "Pon"]


def render():
    st.title("Aspectual class and temporal reading")
    st.caption("The central thesis: telic situations go with anterior/posterior readings, durative ones with "
               "simultaneity (Silvano et al.). Compare the raw counts with the model-adjusted estimate.")

    pt = corpus().query("is_portuguese")
    mode = st.segmented_control("Show", ["Raw counts", "Model-adjusted", "Both"], default="Both")
    clause = st.segmented_control("Clause", ["Main clause", "Participial clause"], default="Main clause")
    if not mode or not clause:
        return
    col, prefix = ("ATMC", "mc") if clause == "Main clause" else ("ATSC", "sc")

    raw = _raw(pt, col).assign(source="Raw counts")
    try:
        adj = aspect_adjusted().query("clause == @prefix").assign(
            **{"class": lambda d: d["aspect_class"].map(ASPECT_NAMES), "source": "Model-adjusted"})
    except Exception as e:
        st.warning(str(e))
        adj = pd.DataFrame()
    parts = {"Raw counts": [raw], "Model-adjusted": [adj], "Both": [raw, adj]}[mode]
    data = pd.concat(parts, ignore_index=True)

    bars = alt.Chart(data).mark_bar().encode(
        x=alt.X("prob:Q", stack="normalize", axis=alt.Axis(format="%"), title=None),
        y=alt.Y("source:N", title=None, sort=["Raw counts", "Model-adjusted"]),
        color=reading_color(),
        order=alt.Order("ord:Q"),
        tooltip=["class:N", "source:N", "reading:N", alt.Tooltip("prob:Q", format=".1%")],
    ).transform_calculate(ord=f"indexof({READINGS!r}, datum.reading)")
    chart = bars.properties(height=70 if mode == "Both" else 40).facet(
        row=alt.Row("class:N", sort=ORDER, title=None, header=alt.Header(labelAngle=0, labelAlign="left")))
    st.altair_chart(chart)

    if mode != "Raw counts" and not adj.empty:
        with st.expander("Model-adjusted estimates with 95% intervals"):
            tbl = adj.assign(estimate=lambda d: d.apply(
                lambda r: f"{r.prob:.0%} [{r.ci_lo:.0%}, {r.ci_hi:.0%}]", axis=1))
            st.dataframe(tbl.pivot(index="class", columns="reading", values="estimate").reindex(ORDER),
                         )

    try:  # widest gap between aspectual classes, per clause, in model-adjusted probability
        adj_all = aspect_adjusted().query("aspect_class != 'Pon'")
        sp = adj_all.groupby(["clause", "reading"])["prob"].agg(lambda s: s.max() - s.min()).groupby("clause").max()
        sc = table("cue_importance.csv").set_index("block").loc["aspect_sc"]
    except Exception:
        sp = None
    st.markdown(
        "**How to read it.** *Raw counts* are the share of clauses the annotators labelled with each reading. "
        "*Model-adjusted* gives every clause the class in turn, keeps its real tense, position and other "
        "clause, and averages B3's predictions: the difference between classes is then the effect of aspect "
        "with the other cues held as they are. Intervals refit the model on resampled sentences.\n\n"
        "**What changes.** Raw, states look anterior; that is tense (many states are present tense). "
        "Adjusted, states lean simultaneous like processes, as the durativity claim predicts. "
        f"The participial clause's classes stay much closer together: at most {sp['sc'] * 100:.0f} points apart, "
        f"against {sp['mc'] * 100:.0f} for the main clause, and removing that block costs {sc.importance_ablation:.3f} "
        f"macro-F1 (95% interval [{sc.ci_lo_ablation:.3f}, {sc.ci_hi_ablation:.3f}])."
    ) if sp is not None else None

import altair as alt
import pandas as pd
import streamlit as st

from charts import counts_by, stacked_share
from data import BLOCK_NAMES, corpus, table


def render():
    st.title("RQ2 · the four Portuguese varieties")
    st.warning("Descriptive only: five separate fits of about 200 sentences. Negative importances mean dropping "
               "a cue helped, which is overfitting. The confirmatory answer needs the pooled model (B6).",
               icon=":material/warning:")
    df = corpus()
    t1, t2, t3 = st.tabs(["Base rates", "Cue ranks", "Performance"])
    with t1:
        st.altair_chart(stacked_share(counts_by(df, "variety"), "variety", sort=["EP", "BP", "AP", "MP", "BE"]))
        cue = st.selectbox("Readings by cue, per variety",
                           ["main aspect", "Position", "connector", "TMC", "participial aspect"])
        v = st.pills("Variety", ["EP", "BP", "AP", "MP", "BE"], default="EP")
        if v:
            st.altair_chart(stacked_share(counts_by(df[df["variety"] == v], cue), cue))
        st.caption("Base rates differ clearly (BP anterior-leaning, English 95% anterior). B6 gives each variety "
                   "its own intercept so these differences are not mistaken for different cue weights.")
    with t2:
        imp = table("variety_cue_importance.csv", index_col=0)
        ranks = table("variety_cue_ranks.csv", index_col=0)
        if imp is None or ranks is None:
            return
        long = imp.reset_index(names="block").melt(id_vars="block", var_name="variety", value_name="importance")
        long["rank"] = long.apply(lambda r: ranks.loc[r["block"], r["variety"]], axis=1)
        long["block"] = long["block"].map(BLOCK_NAMES)
        long["reference"] = long["variety"].eq("BE")
        heat = alt.Chart(long).mark_rect().encode(
            x=alt.X("variety:N", sort=["EP", "BP", "AP", "MP", "BE"], title=None),
            y=alt.Y("block:N", title=None, sort=[BLOCK_NAMES[b] for b in ranks.index]),
            color=alt.Color("importance:Q", scale=alt.Scale(scheme="purpleorange", domainMid=0)),
            opacity=alt.condition("datum.reference", alt.value(0.35), alt.value(1)),
            tooltip=["block", "variety", "rank", alt.Tooltip("importance:Q", format="+.3f")])
        text = heat.mark_text(fontSize=14).encode(text="rank:Q", color=alt.value("#16232A"))
        st.altair_chart((heat + text).properties(height=320))
        st.caption("Cell = rank within the variety; colour = ablation importance. BE faded: reference only.")
    with t3:
        perf = table("variety_performance.csv")
        if perf is not None:
            long = perf.melt(id_vars=["variety", "n"], value_vars=["majority", "logit"],
                             var_name="model", value_name="macro_f1")
            chart = alt.Chart(long).mark_bar().encode(
                x=alt.X("variety:N", sort=["EP", "BP", "AP", "MP"], title=None),
                xOffset="model:N", y=alt.Y("macro_f1:Q", title="macro-F1"),
                color=alt.Color("model:N", scale=alt.Scale(range=["#A8B4B4", "#0E7C7B"]),
                                legend=alt.Legend(orient="top")),
                tooltip=["variety", "model", alt.Tooltip("macro_f1:Q", format=".3f"), "n"])
            st.altair_chart(chart.properties(height=300))
            st.caption("Per-variety logit (3 repeats) against the majority floor.")

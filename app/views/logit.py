import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from charts import reading_color
from data import (ASPECT_NAMES, COLORS, READINGS, corpus, features_for, logit_contributions,
                  table)


def _simulator():
    st.subheader("What-if: build a clause and watch the reading")
    st.caption("B3 fitted on all 793 Portuguese clauses. Change one cue at a time to see its push.")
    pt = corpus().query("is_portuguese")
    tenses = pt["TMC"].value_counts().index.tolist()
    c = st.columns(5)
    atmc = c[0].selectbox("Main-clause aspect", ["Culm", "Pro", "CP", "St", "Pon"],
                          format_func=lambda k: ASPECT_NAMES[k])
    tmc = c[1].selectbox("Main-clause tense", tenses,
                         format_func=lambda t: f"{t} ({(pt['TMC'] == t).sum()})")
    atsc = c[2].selectbox("Participial aspect", ["Culm", "CP", "Pro", "St"],
                          format_func=lambda k: ASPECT_NAMES[k])
    pos = c[3].selectbox("Position", ["Final", "Initial", "Medial"])
    cnt = c[4].toggle("Connector present")

    x = features_for(pos, cnt, atmc, atsc, tmc)
    contrib, scores, classes = logit_contributions(x)
    probs = np.exp(scores - scores.max())
    probs /= probs.sum()
    p = pd.DataFrame({"reading": classes, "score": scores, "p": probs})

    left, right = st.columns([1, 1.4])
    with left:
        st.markdown("**1 · scores → 2 · softmax → probabilities**")
        bars = alt.Chart(p).mark_bar().encode(
            x=alt.X("p:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%"), title=None),
            y=alt.Y("reading:N", sort=READINGS, title=None), color=reading_color(),
            tooltip=["reading", alt.Tooltip("score:Q", format="+.2f"), alt.Tooltip("p:Q", format=".1%")])
        st.altair_chart(bars.properties(height=150))
        st.dataframe(p.set_index("reading").loc[READINGS], column_config={
            "score": st.column_config.NumberColumn(format="%+.2f"),
            "p": st.column_config.NumberColumn("probability", format="%.3f")})
        n_same = ((pt["ATMC"] == atmc) & (pt["TMC"] == tmc) & (pt["Position"] == pos)).sum()
        st.caption(f"{n_same} corpus clauses share this main-clause aspect, tense and position.")
    with right:
        st.markdown("**Which features push which reading** (standardised value × weight)")
        long = contrib.reset_index(names="feature").melt(id_vars="feature", var_name="reading", value_name="push")
        keep = contrib.abs().max(axis=1).sort_values(ascending=False).head(10).index
        long = long[long["feature"].isin(keep)]
        chart = alt.Chart(long).mark_bar().encode(
            y=alt.Y("feature:N", sort=list(keep), title=None),
            x=alt.X("push:Q", title="push on the score"),
            yOffset=alt.YOffset("reading:N", sort=READINGS), color=reading_color(),
            tooltip=["feature", "reading", alt.Tooltip("push:Q", format="+.2f")])
        st.altair_chart(chart.properties(height=420))
    st.caption("Absent features push too: features are standardised, so 'not telic' is informative "
               "because most main clauses are telic.")


def render():
    st.title("B3 · multinomial logistic regression")
    st.caption("One score per reading = starting value + Σ weight × cue; softmax turns the three scores into "
               "probabilities. L2-penalised, features standardised so weights compare across features.")
    coefs = table("logit_coefficients.csv", index_col=0)
    tab1, tab2 = st.tabs(["What-if simulator", "All coefficients"])
    with tab1:
        _simulator()
    with tab2:
        if coefs is None:
            return
        long = coefs.reset_index(names="feature").melt(id_vars="feature", var_name="reading", value_name="w")
        order = coefs.abs().max(axis=1).sort_values(ascending=False).index.tolist()
        heat = alt.Chart(long).mark_rect().encode(
            x=alt.X("reading:N", sort=READINGS, title=None), y=alt.Y("feature:N", sort=order, title=None),
            color=alt.Color("w:Q", scale=alt.Scale(scheme="purpleorange", domainMid=0), title="weight"),
            tooltip=["feature", "reading", alt.Tooltip("w:Q", format="+.3f")])
        text = heat.mark_text(fontSize=12).encode(text=alt.Text("w:Q", format="+.2f"), color=alt.value("#16232A"))
        st.altair_chart((heat + text).properties(height=560, width=420), width="content")
        st.caption("Single columns are not blocks: tense spans seven rows. Rank blocks with the ablation, "
                   "not by reading one row here. Directions: durative → Simul, telic → away from Simul, "
                   "fronted (non-final) → Ant, never Post.")
        st.markdown(f"Colours match the readings used elsewhere: "
                    + " · ".join(f"<span style='color:{COLORS[r]}'>{r}</span>" for r in READINGS),
                    unsafe_allow_html=True)

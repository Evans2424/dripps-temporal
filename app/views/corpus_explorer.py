import altair as alt
import pandas as pd
import streamlit as st

from charts import counts_by, stacked_share
from data import (ASPECT_NAMES, COLORS, ERROR_MODELS, READINGS, corpus, highlight, matrices, oof,
                  predictions, shap_long, tree_path)

SHOW = ["ID", "variety", "TR", "predicted", "confidence", "main aspect", "participial aspect", "TMC",
        "Position", "connector", "Sentence"]
OUTCOMES = ["all", "mispredicted", "correct", "missed by every model", "missed by B3, caught by another"]


def _with_predictions(df: pd.DataFrame, model: str) -> pd.DataFrame:
    """Attach one model's out-of-fold prediction (and B3's confidence) to the corpus."""
    pr = predictions().set_index("ID")
    o = oof(model).set_index("ID")
    return df.assign(
        predicted=df["ID"].map(o["pred"]),
        ok=df["ID"].map(o["pred"] == o["gold"]),
        confidence=df["ID"].map(pr["confidence"]),
        models_wrong=df["ID"].map(pr["models wrong"]),
        b3_ok=df["ID"].map(pr["ok · B3 logit"]),
    )


def _filters(df: pd.DataFrame) -> pd.DataFrame:
    with st.sidebar:
        st.markdown("**Prediction**")
        model = st.selectbox("Model (out of fold)", ERROR_MODELS, key="explorer_model")
        outcome = st.radio("Show", OUTCOMES, key="explorer_outcome",
                           help="'Every model' means all five compared models on the error-analysis page.")
        st.markdown("**Filter the corpus**")
        include_be = st.toggle("Include British English", value=False,
                               disabled=outcome != "all", help="Models are fit on Portuguese only.")
        df = _with_predictions(df, model)
        if not include_be or outcome != "all":
            df = df[df["is_portuguese"]]
        if outcome == "mispredicted":
            df = df[df["ok"] == False]  # noqa: E712 -- NaN for BE rows must not pass
        elif outcome == "correct":
            df = df[df["ok"] == True]  # noqa: E712
        elif outcome == "missed by every model":
            df = df[df["models_wrong"] == len(ERROR_MODELS)]
        elif outcome == "missed by B3, caught by another":
            df = df[(df["b3_ok"] == False) & (df["models_wrong"] < len(ERROR_MODELS))]  # noqa: E712
        pick = lambda label, col: st.multiselect(label, sorted(df[col].dropna().unique()))  # noqa: E731
        sel = {
            "variety": pick("Variety", "variety"),
            "TR": pick("Reading", "TR"),
            "main aspect": pick("Main-clause aspect", "main aspect"),
            "participial aspect": pick("Participial aspect", "participial aspect"),
            "TMC": pick("Main-clause tense", "TMC"),
            "Position": pick("Position", "Position"),
            "connector": pick("Connector", "connector"),
        }
        text = st.text_input("Text contains", placeholder="e.g. de seguida")
    for col, values in sel.items():
        if values:
            df = df[df[col].isin(values)]
    if text:
        df = df[df["Sentence"].str.contains(text, case=False, regex=False)]
    return df


def _sentence_card(row: pd.Series):
    st.markdown(f"#### {row['ID']} · {row['variety']} · annotated **{row['TR']}**")
    st.markdown(f"<div style='font-size:1.15rem;line-height:1.6'>{highlight(row['Sentence'])}</div>",
                unsafe_allow_html=True)
    a = st.columns(5)
    a[0].metric("main aspect", ASPECT_NAMES.get(row["ATMC"], row["ATMC"]))
    a[1].metric("participial aspect", ASPECT_NAMES.get(row["ATSC"], row["ATSC"]))
    a[2].metric("tense", row["TMC"])
    a[3].metric("position", row["Position"])
    a[4].metric("connector", row["connector"])
    if row["n_tendo"] > 1:
        st.caption(f"This sentence contains {row['n_tendo']} auxiliaries; the annotated clause is one of them.")

    if not row["is_portuguese"]:
        st.caption("British English is a reference variety; the models are fit on Portuguese only.")
        return
    df, X, _, _ = matrices()
    idx = int((df["ID"] == row["ID"]).to_numpy().nonzero()[0][0])
    left, mid, right = st.columns([1.1, 1, 1.2])

    with left:
        st.markdown("**B3 out of fold**: predicted by a model that never saw this sentence")
        p = oof("B3 logit").iloc[idx]
        probs = pd.DataFrame({"reading": READINGS, "p": [p[f"p_{r}"] for r in READINGS]})
        chart = alt.Chart(probs).mark_bar().encode(
            x=alt.X("p:Q", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%"), title=None),
            y=alt.Y("reading:N", sort=READINGS, title=None),
            color=alt.Color("reading:N", scale=alt.Scale(domain=READINGS, range=[COLORS[r] for r in READINGS]),
                            legend=None),
            tooltip=["reading", alt.Tooltip("p:Q", format=".1%")],
        ).properties(height=120)
        st.altair_chart(chart)
        verdict = "correct" if p["pred"] == p["gold"] else "wrong"
        st.caption(f"Prediction: **{p['pred']}** ({verdict})")
        st.caption("Every model: " + " · ".join(
            f"{m} **{oof(m).iloc[idx]['pred']}** {'✓' if oof(m).iloc[idx]['pred'] == p['gold'] else '✗'}"
            for m in ERROR_MODELS))

    with mid:
        st.markdown("**The depth-3 tree's questions**")
        st.markdown("  \n".join(f"{i + 1}. {s}" for i, s in enumerate(tree_path(idx))))

    with right:
        st.markdown("**What pushes XGBoost** (TreeSHAP)")
        try:
            s = shap_long("xgboost")
        except Exception as e:  # table missing
            st.caption(str(e))
            return
        pred = st.segmented_control("toward", READINGS, default=row["TR"], key=f"shap_{row['ID']}")
        if not pred:
            return
        s = s[(s["id"] == row["ID"]) & (s["class"] == pred)]
        s = s.assign(value=lambda d: d["feature"].map(X.iloc[idx]))
        s = s.reindex(s["shap"].abs().sort_values(ascending=False).index).head(8)
        chart = alt.Chart(s).mark_bar().encode(
            x=alt.X("shap:Q", title=f"push toward {pred} (log-odds)"),
            y=alt.Y("feature:N", sort=None, title=None),
            color=alt.condition("datum.shap > 0", alt.value(COLORS[pred]), alt.value("#A8B4B4")),
            tooltip=["feature", "value", alt.Tooltip("shap:Q", format="+.3f")],
        ).properties(height=220)
        st.altair_chart(chart)
        st.caption("SHAP describes the model fit on all sentences, not an out-of-fold prediction.")


def render():
    st.title("Corpus explorer")
    st.caption("Filter the annotated clauses, then select one to see its annotation and what each model does with it.")
    df = _filters(corpus())

    c = st.columns([1, 1, 2])
    c[0].metric("clauses matching", len(df))
    scored = df["ok"].dropna()
    if len(scored):
        c[1].metric(f"{st.session_state['explorer_model']} errors", f"{1 - scored.astype(bool).mean():.0%}",
                    help="Share of the matching Portuguese clauses this model mispredicts, out of fold.")
    if len(df):
        mix = df["TR"].value_counts(normalize=True).reindex(READINGS, fill_value=0)
        c[2].markdown("Reading mix: " + " · ".join(
            f"<span style='color:{COLORS[r]};font-weight:600'>{r} {mix[r]:.0%}</span>" for r in READINGS),
            unsafe_allow_html=True)

    event = st.dataframe(df[SHOW], hide_index=True, on_select="rerun", selection_mode="single-row",
                         height=320, column_config={
                             "Sentence": st.column_config.TextColumn(width="large"),
                             "confidence": st.column_config.NumberColumn(
                                 "B3 conf.", format="percent", help="B3's probability for its top reading"),
                         })
    rows = event.selection.rows
    if rows:
        st.divider()
        _sentence_card(df.iloc[rows[0]])
    else:
        st.caption("Select a row to open the sentence.")

    if len(df):
        st.divider()
        st.markdown("**Readings across the current selection**")
        by = st.segmented_control("Break down by", ["variety", "main aspect", "participial aspect", "TMC",
                                                    "Position", "connector", "predicted"], default="variety",
                                 help="'predicted' puts the model's reading on the rows and the annotated "
                                      "reading in the colours: on mispredicted clauses, the confusions.")
        if by:
            st.altair_chart(stacked_share(counts_by(df, by), by))

import altair as alt
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix, f1_score

from charts import interval_dots
from data import READINGS, corpus, model_labels, oof, table

ROLE = {
    "B0 majority": "floor: the base rate alone",
    "B1 one-rule": "floor for a hierarchy claim: the best single cue",
    "B3 logit": "main model: one signed weight per cue and reading",
    "B4 tree(d=4)": "readable cue combinations",
    "B5 forest": "does flexibility find interactions the logit misses?",
    "B5 xgboost": "same question, boosted trees",
    "circular (DR+SR-SC)": "leakage check, not a result",
    "circular control: B3 on same rows": "B3 on the rows the circular bound can use",
}


def render():
    st.title("Model ladder")
    st.caption("Every rung scored the same way: out of fold, sentences kept whole, 10 repeated 5-fold splits, "
               "95% interval from 2,000 resamples of sentences.")
    pt = corpus().query("is_portuguese")
    sizes = pt.groupby("sentence_group").size()
    multi, n_sent = sizes[sizes > 1], len(sizes)
    with st.expander("How every score is measured"):
        st.markdown(
            f"1. **Sentences stay whole.** {len(multi)} Portuguese sentences contain two or more participial clauses "
            f"({multi.sum()} rows) with the same main clause; their rows always go to the same fold, so no sentence is "
            f"tested on something it was trained on. Unit of analysis: {n_sent} sentences.\n"
            "2. **Five folds, rotated.** Train on four, predict the fifth, rotate: every sentence gets one "
            "prediction from a model that never saw it. Each fold keeps the Ant/Post/Simul mix. For the ladder "
            "the whole cut is redone 10 times.\n"
            f"3. **Resample sentences for the interval.** Draw {n_sent} sentences with replacement 2,000 times, "
            "recompute macro-F1, keep the middle 95%. Comparisons are paired: both models on the same draws.\n\n"
            "Fold-to-fold spread is not used: every fold reuses the same sentences, so it is several times "
            "narrower and measures the split, not the data.")
    base = table("baselines.csv")
    if base is not None:
        base = base.assign(role=base["model"].map(ROLE),
                           kind=base["model"].map(lambda m: "leakage check" if "circular" in m else "model"))
        chart = interval_dots(base, y="model", x="macro_f1", lo="ci_lo", hi="ci_hi", color="kind",
                              sort=list(base["model"]), x_title="macro-F1")
        st.altair_chart(chart.properties(height=280))
        st.dataframe(base[["model", "macro_f1", "ci_lo", "ci_hi", "role"]], hide_index=True,
                     column_config={c: st.column_config.NumberColumn(format="%.3f")
                                    for c in ["macro_f1", "ci_lo", "ci_hi"]})

    st.divider()
    st.subheader("Where each model goes wrong")
    label = st.selectbox("Model", model_labels(), index=model_labels().index("B3 logit"))
    o = oof(label)
    left, right = st.columns(2)
    cm = confusion_matrix(o["gold"], o["pred"], labels=READINGS)
    cmdf = (pd.DataFrame(cm, index=READINGS, columns=READINGS).rename_axis("annotated")
            .reset_index().melt(id_vars="annotated", var_name="predicted", value_name="n"))
    cmdf["row_share"] = cmdf["n"] / cmdf.groupby("annotated")["n"].transform("sum")
    heat = alt.Chart(cmdf).mark_rect().encode(
        x=alt.X("predicted:N", sort=READINGS), y=alt.Y("annotated:N", sort=READINGS),
        color=alt.Color("row_share:Q", scale=alt.Scale(scheme="teals"), legend=None),
        tooltip=["annotated", "predicted", "n", alt.Tooltip("row_share:Q", format=".0%")])
    text = heat.mark_text(fontSize=15).encode(
        text="n:Q", color=alt.condition("datum.row_share > 0.5", alt.value("white"), alt.value("#16232A")))
    left.markdown("**Confusion matrix** (one out-of-fold repeat)")
    left.altair_chart((heat + text).properties(height=260))
    f1 = f1_score(o["gold"], o["pred"], labels=READINGS, average=None, zero_division=0)
    right.markdown("**Per-reading F1**")
    right.dataframe(pd.DataFrame({"reading": READINGS, "F1": f1}), hide_index=True,
                    column_config={"F1": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1)})
    right.metric("macro-F1, this repeat", f"{f1.mean():.3f}")
    right.caption("One repeat, so it differs slightly from the 10-repeat figure above.")

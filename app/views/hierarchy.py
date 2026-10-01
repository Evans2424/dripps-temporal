import altair as alt
import pandas as pd
import streamlit as st

from data import BLOCK_NAMES, blocks, table


def render():
    st.title("RQ1 · the cue hierarchy")
    st.caption("Three measures, three questions. Ablation: can we do without the cue? Permutation: does the "
               "fitted model use it? SHAP: which cues drove each prediction?")
    imp = table("cue_importance.csv")
    if imp is None:
        return
    rows = []
    for r in imp.itertuples():
        for m in ("ablation", "permutation"):
            rows.append({"block": BLOCK_NAMES[r.block], "measure": m,
                         "importance": getattr(r, f"importance_{m}"),
                         "lo": getattr(r, f"ci_lo_{m}"), "hi": getattr(r, f"ci_hi_{m}")})
    data = pd.DataFrame(rows).dropna()
    order = [BLOCK_NAMES[b] for b in imp.sort_values("importance_ablation", ascending=False)["block"]]
    colors = alt.Scale(domain=["ablation", "permutation"], range=["#4C4B9E", "#0E7C7B"])
    base = alt.Chart(data).encode(y=alt.Y("block:N", sort=order, title=None),
                                  yOffset=alt.YOffset("measure:N"),
                                  color=alt.Color("measure:N", scale=colors, legend=alt.Legend(orient="top")))
    chart = (
        alt.Chart(pd.DataFrame({"z": [0]})).mark_rule(strokeDash=[4, 4], color="#79888F").encode(x="z:Q")
        + base.mark_rule(strokeWidth=2).encode(x=alt.X("lo:Q", title="drop in macro-F1"), x2="hi:Q")
        + base.mark_circle(size=110, opacity=1).encode(
            x="importance:Q",
            tooltip=["block", "measure", alt.Tooltip("importance:Q", format=".3f"),
                     alt.Tooltip("lo:Q", format=".3f"), alt.Tooltip("hi:Q", format=".3f")])
    ).properties(height=380)
    st.altair_chart(chart)
    crosses = data[(data["measure"] == "ablation") & (data["lo"] <= 0)]["block"].tolist()
    st.markdown(f"**Intervals that include zero (ablation):** {', '.join(crosses)}. "
                "Both measures give the same order; permutation is larger because the model cannot adapt. "
                "The interaction block has no permutation value by design: its columns are products of others.")

    st.divider()
    st.subheader("Does a non-linear model agree?")
    sb = table("shap_blocks.csv")
    if sb is None:
        return
    avg = sb.groupby(["model", "block"])["mean_abs_shap"].mean().reset_index()
    avg["rank"] = avg.groupby("model")["mean_abs_shap"].rank(ascending=False, method="min").astype(int)
    ranks = avg.pivot(index="block", columns="model", values="rank")
    ranks.insert(0, "ablation (B3)", imp.set_index("block")["importance_ablation"].rank(ascending=False, method="min").astype(int))
    ranks.index = ranks.index.map(BLOCK_NAMES)
    ranks = ranks.sort_values("ablation (B3)")

    def mark(col):
        ref = ranks["ablation (B3)"]
        return ["background-color:#E3F0EF" if v == r else "" for v, r in zip(col, ref)]

    st.dataframe(ranks.style.apply(mark, axis=0), )
    st.caption("Shaded: same rank as the ablation. The top two replicate; below that the ensembles lean on "
               "cues that overlap with main-clause aspect. Used is not the same as needed. Compare ranks only: "
               "forest SHAP is in probability units, XGBoost in log-odds.")
    with st.expander("Which features make up each block"):
        st.dataframe(pd.DataFrame([{"block": BLOCK_NAMES[b], "features": ", ".join(f)}
                                   for b, f in blocks().items()]), hide_index=True)

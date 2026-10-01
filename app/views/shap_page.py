import altair as alt
import streamlit as st

from data import COLORS, READINGS, corpus, highlight, matrices, shap_long


def render():
    st.title("TreeSHAP · what the ensembles lean on")
    st.caption(f"Path-dependent TreeSHAP on models fit to all {corpus()['is_portuguese'].sum()} Portuguese clauses: a description of the fitted models, "
               "not an out-of-fold estimate. Compare rankings within a model, never magnitudes across models.")
    c = st.columns(2)
    slug = c[0].segmented_control("Model", ["xgboost", "forest"], default="xgboost",
                                  format_func=lambda s: {"xgboost": "B5 XGBoost", "forest": "B5 forest"}[s])
    reading = c[1].segmented_control("Push toward", READINGS, default="Post")
    if not slug or not reading:
        return
    try:
        s = shap_long(slug)
    except Exception as e:
        st.warning(str(e))
        return
    df, X, _, _ = matrices()
    s = s[s["class"] == reading]
    long_x = X.assign(id=df["ID"].to_numpy()).melt(id_vars="id", var_name="feature", value_name="value")
    s = s.merge(long_x, on=["id", "feature"])
    order = s.groupby("feature")["shap"].apply(lambda v: v.abs().mean()).sort_values(ascending=False)

    left, right = st.columns([1, 1.5])
    with left:
        st.markdown("**Mean |SHAP|**")
        bar = alt.Chart(order.rename("mean_abs").reset_index()).mark_bar(color=COLORS[reading]).encode(
            x=alt.X("mean_abs:Q", title=None), y=alt.Y("feature:N", sort=None, title=None),
            tooltip=["feature", alt.Tooltip("mean_abs:Q", format=".3f")])
        st.altair_chart(bar.properties(height=460))
    with right:
        st.markdown("**Every sentence** (one dot each; colour = does the sentence have the feature?)")
        pick = alt.selection_point(fields=["id"], name="pick")
        strip = alt.Chart(s).mark_circle(size=28, opacity=0.6).encode(
            y=alt.Y("feature:N", sort=list(order.index), title=None),
            x=alt.X("shap:Q", title=f"push toward {reading}"),
            yOffset=alt.YOffset("jitter:Q"),
            color=alt.Color("value:N", scale=alt.Scale(domain=[0, 1], range=["#A8B4B4", COLORS[reading]]),
                            legend=alt.Legend(title="feature present", orient="top")),
            tooltip=["id", "feature", "value", alt.Tooltip("shap:Q", format="+.3f")],
        ).transform_calculate(jitter="random()").add_params(pick).properties(height=460)
        event = st.altair_chart(strip, on_select="rerun", key=f"strip_{slug}_{reading}")

    ids = event.selection.get("pick", []) if event else []
    if ids:
        rid = ids[0]["id"]
        row = corpus().set_index("ID").loc[rid]
        st.markdown(f"**{rid}** · {row['variety']} · annotated **{row['TR']}**")
        st.markdown(highlight(row["Sentence"]), unsafe_allow_html=True)
        one = s[s["id"] == rid].reindex(s[s["id"] == rid]["shap"].abs().sort_values(ascending=False).index)
        st.dataframe(one[["feature", "value", "shap"]].head(8), hide_index=True,
                     column_config={"shap": st.column_config.NumberColumn(format="%+.3f")})
    else:
        st.caption("Click a dot to open that sentence.")

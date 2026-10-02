import streamlit as st

from data import BLOCK_NAMES, corpus, table


def render():
    st.title("Cues to the temporal reading of adverbial perfect participial clauses")
    st.caption("DRIPPS corpus · Portuguese (EP, BP, AP, MP); British English as reference only")

    pt = corpus().query("is_portuguese")
    base, imp = table("baselines.csv"), table("cue_importance.csv")
    c = st.columns(3)
    c[0].metric("Portuguese clauses", len(pt))
    c[1].metric("sentences", pt["sentence_group"].nunique())
    if base is None or imp is None:
        return
    f1 = base.set_index("model")
    b3 = f1.loc["B3 logit"]
    c[2].metric("B3 macro-F1", f"{b3.macro_f1:.3f}", help=f"95% CI [{b3.ci_lo:.3f}, {b3.ci_hi:.3f}]")

    st.subheader("RQ1: cue hierarchy")
    imp = imp.sort_values("importance_ablation", ascending=False)
    st.dataframe(
        imp.assign(cue=imp["block"].map(BLOCK_NAMES))[["cue", "importance_ablation", "ci_lo_ablation", "ci_hi_ablation"]],
        hide_index=True,
        column_config={"cue": "cue block", "importance_ablation": st.column_config.NumberColumn("Δ macro-F1", format="%.3f"),
                       "ci_lo_ablation": st.column_config.NumberColumn("CI low", format="%.3f"),
                       "ci_hi_ablation": st.column_config.NumberColumn("CI high", format="%.3f")})
    st.caption("Drop in out-of-fold macro-F1 when the block is removed and the model refit; "
               "95% CI from resampling sentences.")

    v = imp.set_index("block")
    b6 = table("b6_tests.csv")
    rq2 = ""
    if b6 is not None:
        p_int, p_slope = b6.set_index("term").loc[["intercepts (base rates)", "slopes, all cues"], "p_boot"]
        rq2 = (f"RQ2: base rates {'differ' if p_int < .05 else 'are not shown to differ'} across varieties "
               f"(bootstrap p = {p_int:.3f}); cue weights {'differ' if p_slope < .05 else 'are not shown to differ'} "
               f"(slopes p = {p_slope:.3f}). ")
    ens = f1.loc[["B5 forest", "B5 xgboost"], "macro_f1"]
    st.markdown(
        "- Main-clause aspect and tense carry the reading; clause position is third.\n"
        f"- Participial-clause aspect: Δ = {v.importance_ablation['aspect_sc']:.3f} "
        f"[{v.ci_lo_ablation['aspect_sc']:.3f}, {v.ci_hi_ablation['aspect_sc']:.3f}].\n"
        f"- Random forest ({ens.iloc[0]:.3f}) and XGBoost ({ens.iloc[1]:.3f}) do not exceed the additive logit "
        f"({b3.macro_f1:.3f}): no evidence of cue interactions.\n"
        f"- {rq2}RQ3 (transfer) is not started."
    )

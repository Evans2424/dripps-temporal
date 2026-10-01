import re

import altair as alt
import pandas as pd
import streamlit as st

from charts import counts_by, stacked_share
from data import CUE_COLUMNS, PARTICIPLE, READINGS, corpus, matrices


def _tendo_contexts(sentences: pd.Series) -> pd.DataFrame:
    """What follows each 'tendo': a participle at once, a few words later, or none."""
    part = PARTICIPLE
    rows = []
    for s in sentences:
        for m in re.finditer(r"(?i)\btendo\b(?=((?:\s+\S+){0,4}))", s):
            w = m.group(1).split()
            if w and part.match(w[0]):
                kind = "participle, immediately"
            elif any(part.match(x) for x in w[:4]):
                kind = "participle within 4 words"
            else:
                kind = "no participle (not a clause)"
            rows.append({"what follows": kind, "context": s[max(0, m.start() - 40):m.end() + 40]})
    return pd.DataFrame(rows)


def render():
    st.title("Data exploration")
    st.caption("What the data look like, and the modelling decisions each property forced.")
    df = corpus()
    pt = df[df["is_portuguese"]]

    t1, t2, t3, t4, t5 = st.tabs(["Size and readings", "Cue by cue", "Sparsity", "Finding the clause", "Leakage"])

    with t1:
        c = st.columns(4)
        c[0].metric("Portuguese clauses", len(pt))
        n_var = pt.groupby("variety").size()
        c[1].metric("per variety", f"{n_var.min()}–{n_var.max()}")
        smallest = pd.crosstab(pt["variety"], pt["TR"]).stack().idxmin()
        c[2].metric("smallest reading group", int(pd.crosstab(pt["variety"], pt["TR"]).stack().min()),
                    help=f"{smallest[1]} in {smallest[0]}")
        c[3].metric("test fold, 5 folds", f"≈{len(pt) // 5}")
        normalize = st.toggle("Show shares", value=True, key="t1norm")
        st.altair_chart(stacked_share(counts_by(df, "variety"), "variety", normalize=normalize,
                                      sort=["EP", "BP", "AP", "MP", "BE"]))
        st.markdown(
            "- Posterior leads in EP, AP and MP; **BP is anteriority-dominant**; English is 95% anterior → "
            "**macro-F1, not accuracy**, and English as a reference only.\n"
            f"- About {round(int(pd.crosstab(pt['variety'], pt['TR']).stack().min()) / 5)} {smallest[1]} clauses per "
            f"{smallest[0]} test fold → **per-variety models are noisy**; pool the varieties (B6)."
        )

    with t2:
        cue = st.selectbox("Cue", list(CUE_COLUMNS), format_func=lambda c: CUE_COLUMNS[c])
        col = {"CNT": "connector"}.get(cue, cue)
        facet = st.toggle("Split by variety", value=False)
        data = pt if not st.toggle("Include British English", value=False, key="t2be") else df
        if facet:
            for v in sorted(data["variety"].unique()):
                sub = data[data["variety"] == v]
                st.markdown(f"**{v}** · {len(sub)} clauses")
                st.altair_chart(stacked_share(counts_by(sub, col), col))
        else:
            st.altair_chart(stacked_share(counts_by(data, col), col))
            ct = pd.crosstab(data[col], data["TR"]).reindex(columns=READINGS, fill_value=0)
            ct["n"] = ct.sum(axis=1)
            st.dataframe(ct.sort_values("n", ascending=False))
        st.caption("These are raw counts: a cue's bar mixes its own effect with the cues it co-occurs with. "
                   "Compare a stative main clause by tense to see why a multivariate model is needed.")

    with t3:
        _, X, _, _ = matrices()
        share = X.mean().sort_values().rename("share").reset_index().rename(columns={"index": "feature"})
        chart = alt.Chart(share).mark_bar().encode(
            x=alt.X("share:Q", axis=alt.Axis(format="%"), title="clauses with the feature"),
            y=alt.Y("feature:N", sort=None, title=None),
            color=alt.condition("datum.share < 0.05", alt.value("#8F4E15"), alt.value("#0E7C7B")),
            tooltip=["feature", alt.Tooltip("share:Q", format=".1%")],
        ).properties(height=440)
        rule = alt.Chart(pd.DataFrame({"x": [0.05]})).mark_rule(strokeDash=[4, 4]).encode(x="x:Q")
        left, right = st.columns([1.2, 1])
        left.altair_chart(chart + rule)
        tmc = pt["TMC"].value_counts().rename("clauses").reset_index()
        right.markdown(f"**{len(tmc)} tense labels in Portuguese**, "
                       f"{(tmc['clauses'] < 5).sum()} with fewer than 5 clauses; "
                       f"the top 3 cover {tmc['clauses'].head(3).sum() / len(pt):.0%}.")
        right.dataframe(tmc, hide_index=True, height=360)
        st.caption("Features under 5% (orange) stay in the model but cannot support per-variety estimates: "
                   "another reason for the tense recoding and for pooling.")

    with t4:
        ctx = _tendo_contexts(pt.drop_duplicates("sentence_group")["sentence_norm"])
        summary = ctx["what follows"].value_counts().rename("occurrences").reset_index()
        summary["share"] = (summary["occurrences"] / summary["occurrences"].sum()).map("{:.0%}".format)
        rows = pt.assign(n_rows=pt.groupby("sentence_group")["ID"].transform("size"))
        match = pd.Series(
            ["one row, one tendo" if (r.n_rows == 1 and r.n_tendo == 1) else
             "several rows, one shared tendo" if (r.n_rows > 1 and r.n_tendo == 1) else
             "rows and tendo match in number" if r.n_rows == r.n_tendo else
             "one row, two tendo"
             for r in rows.itertuples()], name="case").value_counts().rename("rows").reset_index()
        a, b = st.columns(2)
        a.markdown("**Is *tendo* a reliable marker?**")
        a.dataframe(summary, hide_index=True)
        b.markdown("**Can each row find its *tendo*?**")
        b.dataframe(match, hide_index=True)
        kind = st.selectbox("Inspect occurrences", ctx["what follows"].unique())
        st.dataframe(ctx[ctx["what follows"] == kind], hide_index=True, height=260)
        st.caption("Participles are found by pattern matching, not by hand: treat the split between the "
                   "first two categories as approximate.")

    with t5:
        st.markdown("Two annotated columns partly **encode the answer**, so no model may use them.")
        c = st.columns(2)
        ann = pt[pt["SR-SC"] != ""]  # the second batch has no SR-SC
        for box, col in zip(c, ["SR-SC", "DR"]):
            ct = pd.crosstab(ann[col], ann["TR"]).reindex(columns=READINGS, fill_value=0)
            box.markdown(f"**{col}** × reading")
            box.dataframe(ct)
        st.caption("*before* is always Ant and *after* always Post; *asynchrony* never co-occurs with Simul. "
                   "The circular baseline shows what these columns alone reach.")

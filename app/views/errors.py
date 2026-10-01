"""Where the models fail, and whether the clause combination explains it.

The additive logit gives the two-clause interactions little weight overall. That
leaves open the linguists' question: are they decisive on the sentences it gets
wrong? Every test here uses the same out-of-fold predictions, so a sentence is
always judged by a model that never saw it.
"""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import binomtest, chi2

from charts import READING_SCALE
from data import (ASPECT_NAMES, ERROR_MODELS, PAIR_MODEL, READINGS, corpus, highlight, matrices, oof,
                  predictions)

CATEGORIES = ["clause combination", "verb meaning (lexical)", "needs wider context",
              "annotation doubtful", "connector / adverbial", "tense", "other"]
INFO = ["variety", "main aspect", "participial aspect", "TMC", "Position", "connector", "Sentence"]


def _frame() -> pd.DataFrame:
    """Predictions joined to the annotation, one row per Portuguese clause."""
    pr = predictions()
    info = corpus().set_index("ID")[INFO + ["ATMC", "ATSC"]]
    df = pr.join(info, on="ID")
    df["wrong"] = ~df["ok · B3 logit"]
    df["pair"] = df["ATMC"] + " × " + df["ATSC"]
    return df


def _error_heatmap(df: pd.DataFrame, ok_col: str):
    cell = (df.assign(wrong=~df[ok_col]).groupby(["main aspect", "participial aspect"])
            .agg(n=("wrong", "size"), errors=("wrong", "sum")).reset_index())
    cell["error rate"] = cell["errors"] / cell["n"]
    sel = alt.selection_point(fields=["main aspect", "participial aspect"], name="cell")
    order = list(ASPECT_NAMES.values())
    base = alt.Chart(cell).encode(
        x=alt.X("participial aspect:N", sort=order, title="participial clause"),
        y=alt.Y("main aspect:N", sort=order, title="main clause"))
    rect = base.mark_rect().encode(
        color=alt.Color("error rate:Q", scale=alt.Scale(scheme="oranges", domain=[0, 1]),
                        legend=alt.Legend(format="%", orient="right")),
        opacity=alt.condition(sel, alt.value(1), alt.value(0.45)),
        tooltip=["main aspect", "participial aspect", "n", "errors",
                 alt.Tooltip("error rate:Q", format=".0%")],
    ).add_params(sel)
    text = base.mark_text(fontSize=13).encode(
        text=alt.Text("label:N"),
        color=alt.condition("datum['error rate'] > 0.5", alt.value("white"), alt.value("#16232A")),
    ).transform_calculate(label="format(datum['error rate'], '.0%') + ' of ' + datum.n")
    return (rect + text).properties(height=300), cell


def _feature_lift(df: pd.DataFrame) -> pd.DataFrame:
    """Error rate with each binary feature on vs off."""
    _, X, _, _ = matrices()
    wrong = df["wrong"].to_numpy()
    rows = []
    for f in X.columns:
        on = X[f].to_numpy() == 1
        if on.sum() < 5 or (~on).sum() < 5:
            continue
        rows.append({"feature": f, "clauses with it": int(on.sum()),
                     "error rate, feature on": wrong[on].mean(), "error rate, off": wrong[~on].mean()})
    out = pd.DataFrame(rows)
    out["difference"] = out["error rate, feature on"] - out["error rate, off"]
    return out.sort_values("difference", ascending=False)


def _cell_fit(df: pd.DataFrame, min_n: int) -> pd.DataFrame:
    """Observed readings vs what each model expects, per aspect-class pair.

    Expected counts are the summed out-of-fold probabilities, so a cell where the
    additive model is systematically off shows up as a gap, whatever its errors.
    """
    probs = {m: oof(m)[[f"p_{r}" for r in READINGS]].to_numpy() for m in ("B3 logit", PAIR_MODEL)}
    rows = []
    for pair, idx in df.groupby("pair").indices.items():
        if len(idx) < min_n:
            continue
        obs = df["gold"].iloc[idx].value_counts().reindex(READINGS, fill_value=0).to_numpy()
        row = {"pair": pair, "n": len(idx)}
        for m, p in probs.items():
            exp = p[idx].sum(axis=0)
            stat = ((obs - exp) ** 2 / np.maximum(exp, 1e-9)).sum()
            tag = "B3" if m == "B3 logit" else "pairs"
            row[f"gap, {tag}"] = 0.5 * np.abs(obs - exp).sum() / len(idx)
            row[f"p, {tag}"] = chi2.sf(stat, df=len(READINGS) - 1)
            for r, e in zip(READINGS, exp):
                row[f"{r} exp. {tag}"] = e / len(idx)
        for r, o in zip(READINGS, obs):
            row[f"{r} observed"] = o / len(idx)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("gap, B3", ascending=False)


def _fixes_breaks(df: pd.DataFrame) -> pd.DataFrame:
    b3 = df["ok · B3 logit"]
    rows = []
    for m in ERROR_MODELS[1:]:
        other = df[f"ok · {m}"]
        fixes, breaks = int((~b3 & other).sum()), int((b3 & ~other).sum())
        p = binomtest(fixes, fixes + breaks).pvalue if fixes + breaks else 1.0
        rows.append({"model": m, "fixes B3 errors": fixes, "breaks B3 successes": breaks,
                     "net": fixes - breaks, "exact McNemar p": p})
    return pd.DataFrame(rows)


def _tab_where(df: pd.DataFrame):
    model = st.segmented_control("Model", ERROR_MODELS, default="B3 logit", key="err_heat_model")
    model = model or "B3 logit"
    chart, _ = _error_heatmap(df, f"ok · {model}")
    left, right = st.columns([1.3, 1])
    with left:
        st.markdown(f"**Error rate by clause combination** · {model}, out of fold")
        event = st.altair_chart(chart, on_select="rerun", key="err_heat")
    picked = event.selection.get("cell") if event and event.selection else None
    with right:
        if picked:
            p = picked[0]
            sub = df[(df["main aspect"] == p["main aspect"]) & (df["participial aspect"] == p["participial aspect"])
                     & ~df[f"ok · {model}"]]
            st.markdown(f"**{p['main aspect']} + {p['participial aspect']}**: {len(sub)} errors")
            st.dataframe(sub.assign(predicted=sub["ID"].map(oof(model).set_index("ID")["pred"]))
                         [["ID", "gold", "predicted", "Sentence"]], hide_index=True, height=300)
        else:
            st.caption("Click a cell to list its mispredicted sentences. Cells with few clauses have "
                       "unstable rates: read the 'of n'.")
    st.divider()
    st.markdown("**Which cue values go with B3's errors?**")
    lift = _feature_lift(df)
    bars = alt.Chart(lift).mark_bar().encode(
        x=alt.X("difference:Q", axis=alt.Axis(format="+%"), title="error rate with the feature − without"),
        y=alt.Y("feature:N", sort=None, title=None),
        color=alt.condition("datum.difference > 0", alt.value("#8F4E15"), alt.value("#0E7C7B")),
        tooltip=["feature", "clauses with it", alt.Tooltip("error rate, feature on:Q", format=".0%"),
                 alt.Tooltip("error rate, off:Q", format=".0%")],
    ).properties(height=420)
    st.altair_chart(bars)
    st.caption("Raw comparisons: a feature can look error-prone because it co-occurs with another one. "
               "Look at `both_telic` and `both_durative`: if the interaction mattered where B3 fails, "
               "clauses that have it would be missed more often.")


def _tab_pairs(df: pd.DataFrame):
    st.markdown(
        "Two tests of the idea that the clause interaction matters **on the sentences B3 gets wrong**. "
        f"The comparison model **{PAIR_MODEL}** is B3 with one indicator for every main × participial "
        "aspect-class pair, so any combination can have its own reading, not only *both telic* and "
        "*both durative*. The trees and XGBoost can build any combination of cues.")
    st.markdown("##### 1 · Does a model that can express combinations rescue B3's errors?")
    fb = _fixes_breaks(df)
    st.dataframe(fb, hide_index=True, column_config={
        "exact McNemar p": st.column_config.NumberColumn(format="%.3f")})
    st.caption(
        "Scoring another model on B3's errors alone would mislead: any different model gets some of them "
        "right by chance. What counts is the balance: errors fixed against successes lost, on the same "
        "sentences and folds. If the interaction were the missing piece, fixes would clearly outnumber breaks.")
    model = st.selectbox("Where does this model's rescue happen?", ERROR_MODELS[1:], key="err_rescue")
    ok = df[f"ok · {model}"]
    moved = pd.DataFrame({
        "fixed": (~df["ok · B3 logit"] & ok).groupby(df["pair"]).sum(),
        "broken": (df["ok · B3 logit"] & ~ok).groupby(df["pair"]).sum(),
        "n": df.groupby("pair").size()})
    moved = moved[(moved["fixed"] + moved["broken"]) > 0].assign(net=lambda d: d["fixed"] - d["broken"])
    st.dataframe(moved.sort_values("net", ascending=False), height=240)

    st.markdown("##### 2 · Is the additive model systematically off in any clause combination?")
    min_n = st.slider("Only combinations with at least n clauses", 5, 60, 15, key="err_min_n")
    fit = _cell_fit(df, min_n)
    long = []
    for _, r in fit.iterrows():
        for rd in READINGS:
            long += [{"pair": r["pair"], "reading": rd, "source": "observed", "share": r[f"{rd} observed"]},
                     {"pair": r["pair"], "reading": rd, "source": "B3 expects", "share": r[f"{rd} exp. B3"]},
                     {"pair": r["pair"], "reading": rd, "source": "pairs model expects",
                      "share": r[f"{rd} exp. pairs"]}]
    chart = alt.Chart(pd.DataFrame(long)).mark_point(filled=True, size=90).encode(
        x=alt.X("share:Q", axis=alt.Axis(format="%"), scale=alt.Scale(domain=[0, 1]), title=None),
        y=alt.Y("reading:N", sort=READINGS, title=None),
        color=alt.Color("reading:N", scale=READING_SCALE, legend=None),
        shape=alt.Shape("source:N", scale=alt.Scale(range=["circle", "triangle-left", "cross"]),
                        legend=alt.Legend(orient="top", title=None)),
        opacity=alt.condition("datum.source == 'observed'", alt.value(1), alt.value(0.7)),
        tooltip=["pair", "reading", "source", alt.Tooltip("share:Q", format=".0%")],
    ).properties(height=80, width=260).facet(
        facet=alt.Facet("pair:N", sort=list(fit["pair"]), title=None), columns=3)
    st.altair_chart(chart)
    show = fit[["pair", "n", "gap, B3", "p, B3", "gap, pairs", "p, pairs"]]
    st.dataframe(show, hide_index=True, column_config={
        c: st.column_config.NumberColumn(format="%.3f") for c in ["gap, B3", "p, B3", "gap, pairs", "p, pairs"]})
    st.info(
        "How to read the two tests together: if the pairs model matches each combination's reading mix "
        "but still fixes no errors, the combination sets the odds inside the cell and not which "
        "reading a given sentence takes. The errors are then the cell's minority readings. "
        "The next tab lists them next to clauses with identical annotation.", icon=":material/lightbulb:")
    st.caption(
        "Sorted by B3's gap: the share of clauses whose reading would have to change for B3's average "
        "prediction to match the annotation. p is a χ² goodness-of-fit test of observed against expected "
        f"counts (2 df), one per row, uncorrected for the {len(fit)} tests. Codes: "
        + ", ".join(f"{k} = {v}" for k, v in ASPECT_NAMES.items()) + ".")


def _tab_confidence(df: pd.DataFrame):
    hist = alt.Chart(df.assign(outcome=np.where(df["wrong"], "wrong", "right"))).mark_bar(opacity=0.85).encode(
        x=alt.X("p_gold:Q", bin=alt.Bin(step=0.05), axis=alt.Axis(format="%"),
                title="B3's probability for the annotated reading"),
        y=alt.Y("count():Q", title="clauses", stack=None),
        color=alt.Color("outcome:N", scale=alt.Scale(domain=["right", "wrong"], range=["#0E7C7B", "#8F4E15"]),
                        legend=alt.Legend(orient="top", title=None)),
    ).properties(height=260)
    st.altair_chart(hist)
    thr = st.slider("Call an error 'confident' when B3's top reading has at least", 0.4, 0.9, 0.6, 0.05,
                    format="%.2f", key="err_thr")
    err = df[df["wrong"]]
    conf = err[err["confidence"] >= thr]
    near = err[err["margin"] < 0.1]
    c = st.columns(3)
    c[0].metric("B3 errors", len(err))
    c[1].metric("confidently wrong", len(conf), help="The cues point firmly elsewhere.")
    c[2].metric("near misses", len(near), help="Top two readings within 10 points: the cues are split.")
    st.markdown(
        "- **Confidently wrong**: the annotated cues point firmly to another reading. Either something "
        "outside the annotation decides the reading (the verbs, context, world knowledge), or the annotation "
        "deserves a second look.\n"
        "- **Near misses**: the cues are genuinely split. These are where a combination effect, if there is "
        "one, has the best chance to tip the balance.")
    kind = st.segmented_control("List", ["confidently wrong", "near misses"], default="confidently wrong",
                                key="err_kind")
    sub = conf if kind != "near misses" else near
    st.dataframe(sub.sort_values("confidence", ascending=False)
                 [["ID", "gold", "pred", "confidence", "p_gold", "models wrong", "pair", "TMC", "Sentence"]],
                 hide_index=True, height=320, column_config={
                     "confidence": st.column_config.NumberColumn(format="percent"),
                     "p_gold": st.column_config.NumberColumn("p(annotated)", format="percent"),
                     "Sentence": st.column_config.TextColumn(width="large")})


def _tab_hard(df: pd.DataFrame):
    counts = df["models wrong"].value_counts().reindex(range(len(ERROR_MODELS) + 1), fill_value=0)
    st.altair_chart(alt.Chart(counts.rename("clauses").reset_index().rename(columns={"index": "models wrong"}))
                    .mark_bar(color="#4C4B9E").encode(
                        x=alt.X("models wrong:O", title=f"models (of {len(ERROR_MODELS)}) that mispredict it"),
                        y="clauses:Q", tooltip=["models wrong", "clauses"]).properties(height=220))
    hard = df[df["models wrong"] == len(ERROR_MODELS)]
    st.markdown(f"**{len(hard)} clauses every model misses.** No combination of the annotated cues gets them "
                "right, so whatever decides their reading is not in the annotation: the prime candidates for "
                "close reading.")
    by = st.segmented_control("Compare with all clauses by", ["pair", "variety", "TMC", "gold", "Position"],
                              default="pair", key="err_hard_by")
    if by:
        cmp_ = pd.DataFrame({"hard": hard[by].value_counts(normalize=True),
                             "all": df[by].value_counts(normalize=True)}).fillna(0)
        cmp_ = cmp_.assign(over=cmp_["hard"] - cmp_["all"]).sort_values("over", ascending=False)
        st.dataframe(cmp_, column_config={c: st.column_config.NumberColumn(format="percent")
                                          for c in ["hard", "all", "over"]}, height=240)
    st.dataframe(hard[["ID", "variety", "gold", "pred", "pair", "TMC", "Position", "connector", "Sentence"]],
                 hide_index=True, height=320,
                 column_config={"Sentence": st.column_config.TextColumn(width="large")})


def _profiles(df: pd.DataFrame) -> pd.Series:
    """A key per clause that is identical exactly when the model inputs are."""
    _, X, _, _ = matrices()
    return pd.Series(X.astype(str).agg("".join, axis=1).to_numpy(), index=df.index)


def _tab_twins(df: pd.DataFrame):
    key = _profiles(df)
    ct = pd.crosstab(key, df["gold"])
    ceiling = ct.max(axis=1).sum() / len(df)
    modal = key.map(ct.idxmax(axis=1))
    hard = df["models wrong"] == len(ERROR_MODELS)
    c = st.columns(3)
    c[0].metric("distinct cue profiles", len(ct), help="Clauses with the same values on every model input.")
    c[1].metric("best possible accuracy from the cues", f"{ceiling:.0%}",
                help="Predict each profile's most common reading, scored on the same data: an optimistic "
                     f"ceiling for any model of these cues. B3 reaches {(~df['wrong']).mean():.0%} out of fold.")
    c[2].metric("every-model misses that are a minority reading of their profile",
                f"{(df.loc[hard, 'gold'] != modal[hard]).mean():.0%}")
    st.markdown(
        "Two clauses with **identical annotation** cannot be told apart by any model of these cues, however "
        "it combines them. When they differ in reading, the difference lies outside the annotation. These "
        "sets are minimal pairs for close reading: what does the minority sentence have that the others lack?")
    mixed = ct[(ct > 0).sum(axis=1) > 1]
    stats = pd.DataFrame({
        "clauses": mixed.sum(axis=1),
        "B3 errors": df["wrong"].groupby(key).sum().reindex(mixed.index),
        "readings": [" · ".join(f"{r} {int(n)}" for r, n in row.items() if n) for _, row in mixed.iterrows()],
        "profile": [_describe(df[key == k].iloc[0], df[key == k]) for k in mixed.index],
    }).sort_values("B3 errors", ascending=False)
    pick = st.selectbox("Profile (most errors first)", stats.index,
                        format_func=lambda k: f"{stats.at[k, 'profile']}  —  {stats.at[k, 'readings']}",
                        key="err_twin")
    sub = df[key == pick]
    cols = st.columns(len(READINGS))
    for col, r in zip(cols, READINGS):
        part = sub[sub["gold"] == r]
        col.markdown(f"**{r}** · {len(part)}" + ("  (the model's choice)" if r == modal[sub.index[0]] else ""))
        for _, row in part.head(40).iterrows():
            col.markdown(f"<div style='font-size:.9rem;margin-bottom:.6rem'><code>{row['ID']}</code> "
                         f"{highlight(row['Sentence'])}</div>", unsafe_allow_html=True)
        if len(part) > 40:
            col.caption(f"… and {len(part) - 40} more")


def _describe(row: pd.Series, group: pd.DataFrame) -> str:
    tenses = "/".join(sorted(group["TMC"].unique()))
    pos = "non-final" if row["Position"] != "Final" else "final"
    cnt = "connector" if row["connector"] != "none" else "no connector"
    return f"{row['pair']} · {tenses} · {pos} · {cnt}"


def _tab_lexical(df: pd.DataFrame):
    st.markdown("The annotation records the verbs' aspectual class, not the verbs themselves. If particular "
                "participles are missed again and again, the verb's own meaning may carry information the "
                "class does not.")
    min_n = st.slider("Participles seen at least", 2, 15, 4, key="err_lex_n")
    lex = (df[df["participle"] != ""].groupby("participle")
           .agg(n=("wrong", "size"), errors=("wrong", "sum"),
                readings=("gold", lambda s: " ".join(f"{r}:{(s == r).sum()}" for r in READINGS if (s == r).any())))
           .query("n >= @min_n"))
    lex["error rate"] = lex["errors"] / lex["n"]
    st.dataframe(lex.sort_values(["error rate", "n"], ascending=False), height=360, column_config={
        "error rate": st.column_config.ProgressColumn(format="percent", min_value=0, max_value=1)})
    missing = (df["participle"] == "").mean()
    st.caption(f"Participles are found by pattern matching after *tendo* ({missing:.0%} of clauses without a "
               "match). In sentences with two *tendo*, the first participle is taken, which may not be the "
               "annotated clause.")


def _tab_review(df: pd.DataFrame):
    st.markdown("A worksheet for reading the errors. Tag each one with what you think decides its reading, "
                "then download the sheet. Nothing is saved on the server: **download before closing the tab**.")
    c = st.columns(3)
    gold = c[0].multiselect("Annotated", READINGS, key="rev_gold")
    pred = c[1].multiselect("B3 predicted", READINGS, key="rev_pred")
    scope = c[2].radio("Errors of", ["B3", "every model"], horizontal=True, key="rev_scope")
    err = df[df["wrong"]] if scope == "B3" else df[df["models wrong"] == len(ERROR_MODELS)]
    if gold:
        err = err[err["gold"].isin(gold)]
    if pred:
        err = err[err["pred"].isin(pred)]
    sheet = err[["ID", "variety", "gold", "pred", "confidence", "models wrong", "pair", "TMC", "Sentence"]].assign(
        category=pd.Series(dtype="object"), note="")
    edited = st.data_editor(sheet, hide_index=True, height=420, key="rev_sheet",
                            disabled=[c for c in sheet.columns if c not in ("category", "note")],
                            column_config={
                                "category": st.column_config.SelectboxColumn(options=CATEGORIES),
                                "note": st.column_config.TextColumn(width="medium"),
                                "confidence": st.column_config.NumberColumn(format="percent"),
                                "Sentence": st.column_config.TextColumn(width="large")})
    tagged = edited["category"].notna().sum()
    st.caption(f"{len(edited)} errors listed, {tagged} tagged.")
    if tagged:
        st.bar_chart(edited["category"].value_counts(), horizontal=True, height=200)
    st.download_button("Download the worksheet (CSV)", edited.to_csv(index=False).encode("utf-8"),
                       file_name="dripps_error_review.csv", mime="text/csv")


def render():
    st.title("Error analysis")
    st.caption("Which sentences the models get wrong, and whether the combination of the two clauses' "
               "aspect explains them. All predictions are out of fold (one repeat, sentence-grouped).")
    df = _frame()
    c = st.columns(4)
    c[0].metric("B3 errors", f"{df['wrong'].sum()} of {len(df)}")
    fb = _fixes_breaks(df).set_index("model").loc[PAIR_MODEL]
    c[1].metric(f"{PAIR_MODEL}: fixed − broken", f"{int(fb['fixes B3 errors'])} − {int(fb['breaks B3 successes'])}",
                help="B3 errors it gets right, against B3 successes it gets wrong.")
    c[2].metric("missed by every model", int((df["models wrong"] == len(ERROR_MODELS)).sum()))
    c[3].metric("B3 confidently wrong (≥60%)", int((df["wrong"] & (df["confidence"] >= 0.6)).sum()))

    tabs = st.tabs(["Where the errors are", "Clause combination on the errors", "Same cues, other reading",
                    "Confidence", "Missed by every model", "Participles", "Review worksheet"])
    for tab, fn in zip(tabs, [_tab_where, _tab_pairs, _tab_twins, _tab_confidence, _tab_hard, _tab_lexical,
                              _tab_review]):
        with tab:
            fn(df)

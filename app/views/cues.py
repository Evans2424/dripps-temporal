"""Cue by cue: the corpus evidence, what every method says about the cue, and the linguistic conclusion it supports.

Numbers are read from the corpus and results/tables, so they follow re-runs; the claim, status and reading of each
cue are written by hand and must be revisited when a status changes (e.g. after annotator agreement)."""

import numpy as np
import pandas as pd
import streamlit as st

from data import BLOCK_NAMES, READINGS, blocks, matrices, table

STATUS = {"supported": ":green[**supported**]", "new": ":blue[**new finding**]",
          "qualified": ":orange[**supported, qualified**]", "inconclusive": ":gray[**inconclusive**]",
          "not supported": ":red[**not supported**]", "candidate": ":violet[**candidate (untested)**]"}

#: block -> (claim from the literature, status, linguistic reading, encoder/LLM perturbation row or None)
CARDS = {
    "aspect_mc": (
        "Durative situations go with simultaneity, telic ones with anteriority or posteriority (Silvano et al. 2021).",
        "supported",
        "The main event's aspectual class sets the temporal frame, and the participial clause is read relative to it: "
        "the participial clause behaves as temporally anaphoric on the main clause. This is the strongest cue for "
        "every method that reads the annotated cues.",
        "mask_main_verb - ctrl_main_verb (d_P_true)"),
    "tense_mc": (
        "No strong prior claim about main-clause tense for this construction.",
        "new",
        "A present-tense main clause pulls the reading to Ant. A plausible reading (an interpretation, not tested): "
        "a present main clause describes a current situation, and a perfect participial clause then reports what "
        "led up to it. Tense and aspect live on the same verb, so masking the main verb removes both.",
        "mask_main_verb - ctrl_main_verb (d_P_true)"),
    "position": (
        "Non-final (fronted) participial clauses read as anterior (Lobo 2003).",
        "qualified",
        "Near-categorical but rare: fronting all but blocks the posterior reading. That fits an order-of-mention "
        "(iconicity) principle: what is said first happened first. It ranks only third because ablation weights a "
        "cue by how often it applies. Every text model shifts towards Ant when the clause is moved to the front, but "
        "the moved sentences keep their original label, so that shift is not yet an error measure.",
        "move_apc_initial"),
    "connector": (
        "Connectors (in Portuguese, almost always *mesmo*) shape the relation between the clauses.",
        "inconclusive",
        "Too rare to decide: about 45 clauses, most of them BP, and collinear with position (often fronted). Its "
        "ablation interval includes zero.",
        None),
    "interaction": (
        "Accounts that require the same aspectual configuration (both telic, or both durative) in both clauses.",
        "not supported",
        "The additive model loses nothing without the both-clause terms, and tree ensembles that could exploit "
        "interactions gain nothing. The cues combine additively: no configuration across both clauses is needed.",
        None),
    "aspect_sc": (
        "The participial clause's own aspectual class contributes to its reading.",
        "not supported",
        "Dispensable given the other cues: removing it, and with it every both-clause term, costs nothing. Two "
        "artefacts are ruled out by the checks on this card. It is not near-constant: it varies widely, and on its "
        "own it is associated with the reading. Nor is it a copy of main-clause aspect: the two are only weakly "
        "correlated. Its signal is real but already carried by the other cues. **Still open (panel review):** "
        "annotator agreement on how it was coded.",
        "mask_participle - ctrl_participle (d_P_true)"),
}

BEYOND = [
    ("Explicit temporal adverbials (*depois*, *já*, *após*, *há 3 anos*…)", "inconclusive",
     "Not annotated, but in the text. Masking them changes encoder and LLM predictions little beyond random-word "
     "controls, and they are only slightly more frequent where Jev beats B3.",
     "mask_adverbials - ctrl_adverbials (d_P_true)"),
    ("Lexical meaning of the verbs, and the timeline (sequences of dates, world knowledge)", "candidate",
     "The best current explanation of what Jev adds. Among clauses with identical annotated cues it still separates "
     "the readings, and most of the clauses it wins are Post/Simul confusions that need the events' meaning. Evidence "
     "is descriptive. Tests B2, C2 and C3 in `docs/T5_ROBUSTNESS_PLAN.md` decide it.",
     None),
]


def f(x, d=3, signed=True):
    """Effects and interval ends carry their sign (+0.580, -0.033); p-values (signed=False) don't."""
    if x is None or pd.isna(x):
        return "—"
    return f"{x:+.{d}f}" if signed and x != 0 else f"{x:.{d}f}"


def _corpus(cols):
    df, X, y, _ = matrices()
    base = y.value_counts(normalize=True).reindex(READINGS)
    rows = [{"cue present": "all clauses (base rate)", "n": len(y), **(100 * base).round(0).to_dict()}]
    for c in cols:
        m = X[c].to_numpy() == 1
        if m.sum():
            share = y[m].value_counts(normalize=True).reindex(READINGS, fill_value=0)
            rows.append({"cue present": c, "n": int(m.sum()), **(100 * share).round(0).to_dict()})
    return pd.DataFrame(rows)


def _shap_ranks():
    s = table("shap_blocks.csv")
    if s is None:
        return {}
    tot = s.groupby(["model", "block"]).mean_abs_shap.sum()
    return {m: tot[m].rank(ascending=False).astype(int).to_dict() for m in tot.index.levels[0]}


def _text_models(row):
    out = []
    t3 = table("t3_ablation.csv")
    if t3 is not None and row in set(t3.perturbation):
        r = t3.set_index("perturbation").loc[row]
        out.append(("encoder (xlm-r, plain)", r))
    t5 = table("t5_decision_perturb.csv")
    if t5 is not None:
        for m, d in t5[t5.perturbation == row].groupby("model", sort=False):
            out.append((m, d.iloc[0]))
    return out


def _card(key, shap):
    claim, status, reading, row = CARDS[key]
    cols = blocks()[key]
    cue = table("cue_importance.csv")
    coef = table("logit_coefficients.csv")
    b6 = table("b6_tests.csv")
    diag = table("t5_decision_diag.csv")
    c = cue.set_index("block").loc[key] if cue is not None else None

    st.subheader(BLOCK_NAMES[key])
    st.markdown(f"**Claim tested:** {claim}  \n**Status:** {STATUS[status]}")
    left, right = st.columns(2)
    with left:
        st.markdown("**Corpus: reading (%) when the cue is present**")
        st.dataframe(_corpus(cols), hide_index=True)
        if coef is not None:
            st.markdown("**B3 coefficients** (standardised; + pushes towards the reading)")
            st.dataframe(coef.set_index(coef.columns[0]).loc[cols].round(2), height=38 + 35 * len(cols))
    with right:
        st.markdown("**How much the models need it**")
        lines = []
        if c is not None:
            lines.append(f"- Refit ablation: **{f(c.importance_ablation)}** [{f(c.ci_lo_ablation)}, {f(c.ci_hi_ablation)}]"
                         + (" — *interval includes 0*" if c.ci_lo_ablation <= 0 <= c.ci_hi_ablation else ""))
            if not pd.isna(c.importance_permutation):
                lines.append(f"- Permutation: {f(c.importance_permutation)}")
        if shap:
            lines.append("- SHAP rank: " + ", ".join(f"{m.replace('B5 ', '')} {r.get(key, '—')}/6" for m, r in shap.items()))
        if b6 is not None:
            r = b6.set_index("term").loc[f"slopes, {key}"]
            lines.append(f"- Differs by variety (B6 slopes): p = {f(r.p_boot, signed=False)}, Holm {f(r.p_boot_holm, signed=False)}")
        st.markdown("\n".join(lines))
        tm = _text_models(row) if row else []
        if tm:
            what = "shift in P(Ant) when the clause is fronted" if row == "move_apc_initial" else \
                "change in P(true reading) when masked, minus the random-word control"
            st.markdown(f"**Text models:** {what}")
            k = "d_P_Ant" if row == "move_apc_initial" else "d_P_true"
            st.dataframe(pd.DataFrame([{"model": m, "effect": r[k], "95% CI": f"[{f(r[k + '_lo'])}, {f(r[k + '_hi'])}]",
                                        "rows": int(r.n_rows)} for m, r in tm]).round(3), hide_index=True)
        if diag is not None:
            g = diag[diag.metric.str.contains(f"share of \\|coef\\| in {key},", regex=True)]
            if len(g):
                gold = g[g.metric.str.endswith("gold")].value.iloc[0]
                llm = g[g.metric.str.endswith("LLM")].set_index("model").value
                st.markdown("**Cue strength vs cue validity** — share of B3's weight on this cue when refit on each "
                            f"model's answers (gold labels: **{gold:.0%}**): "
                            + ", ".join(f"{m} {v:.0%}" for m, v in llm.items()))
    if key == "aspect_sc":  # the panel's artefact checks: does it vary, and is it just a copy of main-clause aspect?
        _, X, _, _ = matrices()
        st.markdown("**Artefact check: participial vs main-clause aspect** — " + "; ".join(
            f"{a}: same value in both clauses {np.mean(X['mc_' + a] == X['sc_' + a]):.0%}, "
            f"r = {np.corrcoef(X['mc_' + a], X['sc_' + a])[0, 1]:.2f}" for a in ("dynamic", "durative", "telic")))
    st.markdown(f"**Linguistic reading.** {reading}")
    st.divider()


def render():
    st.title("Cue by cue: evidence and linguistic conclusions")
    st.caption("One card per annotated cue block: what the corpus shows, how much each model family needs the cue, and "
               "the conclusion it supports. Numbers are live (out-of-fold, 95% sentence-bootstrap intervals); claims and "
               "statuses are hand-written and tentative until annotator agreement is measured.")
    shap = _shap_ranks()
    cue = table("cue_importance.csv")
    order = cue.sort_values("importance_ablation", ascending=False).block.tolist() if cue is not None else list(CARDS)
    st.markdown("**Ranking by refit ablation:** " + " → ".join(BLOCK_NAMES[b] for b in order))
    tabs = st.tabs([BLOCK_NAMES[b] for b in order] + ["Beyond the inventory"])
    for t, b in zip(tabs, order):
        with t:
            _card(b, shap)
    with tabs[-1]:
        st.caption("Factors the annotation does not code, which only the text models can use.")
        for title, status, reading, row in BEYOND:
            st.subheader(title)
            st.markdown(f"**Status:** {STATUS[status]}\n\n{reading}")
            tm = _text_models(row) if row else []
            if tm:
                st.dataframe(pd.DataFrame([{"model": m, "effect on P(true)": r.d_P_true,
                                            "95% CI": f"[{f(r.d_P_true_lo)}, {f(r.d_P_true_hi)}]", "rows": int(r.n_rows)}
                                           for m, r in tm]).round(3), hide_index=True)
            st.divider()

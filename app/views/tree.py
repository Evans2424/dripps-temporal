import pandas as pd
import streamlit as st

from data import COLORS, READINGS, corpus, tree_structure

QUESTION = {
    "mc_durative": "Main clause durative?", "mc_telic": "Main clause telic?",
    "mc_dynamic": "Main clause dynamic?", "tense_present": "Present-tense main clause?",
    "pos_nonfinal": "Participial clause fronted?", "sc_durative": "Participial clause durative?",
    "sc_telic": "Participial clause telic?", "sc_dynamic": "Participial clause dynamic?",
    "has_connector": "Connector present?", "mc_perfective": "Perfective main clause?",
    "mc_perfect": "Perfect main clause?", "both_telic": "Both clauses telic?",
    "both_durative": "Both clauses durative?",
}


def _dot(nodes: list[dict], selected: int | None) -> str:
    lines = ['digraph T {', 'rankdir=TB; node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=11];',
             'edge [fontname="Helvetica", fontsize=10];']
    for n in nodes:
        share = max(n["counts"]) / n["n_samples"]
        head = f"leaf {n['id']}" if n["is_leaf"] else QUESTION.get(n["feature"], n["feature"])
        label = (f"{head}\\nn={n['n_samples']} · {n['predicted']} {share:.0%}\\n"
                 + " / ".join(f"{c}" for c in n["counts"]))
        fill = COLORS[n["predicted"]] + ("FF" if n["is_leaf"] else "33")
        font = "white" if n["is_leaf"] else "#16232A"
        pen = ', penwidth=4, color="#E0913F"' if n["id"] == selected else ""
        lines.append(f'{n["id"]} [label="{label}", fillcolor="{fill}", fontcolor="{font}"{pen}];')
        if not n["is_leaf"]:
            lines.append(f'{n["id"]} -> {n["left"]} [label="no"];')
            lines.append(f'{n["id"]} -> {n["right"]} [label="yes"];')
    lines.append("}")
    return "\n".join(lines)


def render():
    st.title("B4 · the decision tree, read as rules")
    st.caption(f"Depth-3 tree fit on all {corpus()['is_portuguese'].sum()} Portuguese clauses: each box asks one yes/no question about one "
               "cue. Counts are Ant / Post / Simul. Pick a leaf to read the sentences it covers.")
    tree = tree_structure()
    nodes = tree["nodes"]
    leaves = [n for n in nodes if n["is_leaf"]]
    leaf_id = st.pills("Leaf", [n["id"] for n in leaves],
                       format_func=lambda i: f"leaf {i} · {nodes[i]['predicted']} · n={nodes[i]['n_samples']}")
    st.graphviz_chart(_dot(nodes, leaf_id))

    if leaf_id is not None:
        leaf = nodes[leaf_id]
        df = corpus().set_index("ID").loc[leaf["rows"]].reset_index()
        right = (df["TR"] == leaf["predicted"]).mean()
        st.markdown(f"**Leaf {leaf_id}** predicts **{leaf['predicted']}**: right for {right:.0%} of its "
                    f"{len(df)} sentences.")
        mix = df["TR"].value_counts().reindex(READINGS, fill_value=0)
        st.markdown(" · ".join(f"<span style='color:{COLORS[r]};font-weight:600'>{r} {mix[r]}</span>"
                               for r in READINGS), unsafe_allow_html=True)
        only_wrong = st.toggle("Only sentences the leaf gets wrong")
        if only_wrong:
            df = df[df["TR"] != leaf["predicted"]]
        st.dataframe(df[["ID", "variety", "TR", "main aspect", "TMC", "Position", "Sentence"]],
                     hide_index=True, height=320)
    st.caption("No split uses the participial clause: every split is on the main clause or on position.")

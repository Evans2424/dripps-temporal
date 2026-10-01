"""Interactive explorer for the DRIPPS cue-hierarchy results.

    make app          # or: .venv/bin/streamlit run app/streamlit_app.py

Reads the corpus and results/tables/ (run `make all` first) and refits the cheap
models on demand. The Standalone viewer page embeds experiments/06_viewer.py's output.
"""

import streamlit as st

from views import (aspect, corpus_explorer, errors, exploration, hierarchy, logit, methods,
                   models_page, overview, shap_page, tree, varieties, viewer)

st.set_page_config(page_title="DRIPPS cue hierarchy", page_icon=":material/schedule:", layout="wide")

pages = {
    "Start": [
        st.Page(overview.render, title="Overview", icon=":material/home:", url_path="overview", default=True),
        st.Page(corpus_explorer.render, title="Corpus explorer", icon=":material/search:", url_path="corpus"),
    ],
    "Data": [
        st.Page(exploration.render, title="Data exploration", icon=":material/bar_chart:", url_path="data"),
    ],
    "Models": [
        st.Page(models_page.render, title="Model ladder", icon=":material/stairs:", url_path="ladder"),
        st.Page(logit.render, title="B3 logit and what-if", icon=":material/tune:", url_path="logit"),
        st.Page(tree.render, title="B4 decision tree", icon=":material/account_tree:", url_path="tree"),
    ],
    "Results": [
        st.Page(hierarchy.render, title="Cue hierarchy (RQ1)", icon=":material/leaderboard:", url_path="hierarchy"),
        st.Page(aspect.render, title="Aspect and reading", icon=":material/timeline:", url_path="aspect"),
        st.Page(shap_page.render, title="TreeSHAP", icon=":material/scatter_plot:", url_path="shap"),
        st.Page(varieties.render, title="Varieties (RQ2)", icon=":material/public:", url_path="varieties"),
        st.Page(errors.render, title="Error analysis", icon=":material/error_med:", url_path="errors"),
    ],
    "Reference": [
        st.Page(methods.render, title="Methods and references", icon=":material/menu_book:", url_path="methods"),
        st.Page(viewer.render, title="Standalone viewer", icon=":material/dashboard:", url_path="viewer"),
    ],
}

st.navigation(pages).run()

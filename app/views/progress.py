import pandas as pd
import streamlit as st

from data import corpus, results
from dripps import progress


def render():
    st.title("Project summary")
    st.caption("What has been run, what it shows so far, what is running, and what comes next. Scores and intervals are read "
               "from `results/tables`; intervals are 95%, from resampling sentence groups.")
    try:
        s = progress.summary(len(corpus().query("is_portuguese")))
    except results.MissingTable as e:
        st.warning(str(e))
        return

    def show(rows, cols):
        st.table(pd.DataFrame(rows).rename(columns=cols).set_index(next(iter(cols.values()))))

    st.subheader(":green[Completed]")
    show(s["done"], {"name": "Experiment", "setup": "What was run", "result": "Result", "conclusion": "Conclusion",
                  "direction": "Linguistic direction"})
    st.subheader(":orange[Running]")
    st.caption("Exploratory: added after the decision rules were fixed, so no single interval is evidence on its own.")
    show(s["running"], {"name": "Experiment", "setup": "Design", "result": "Comparison", "conclusion": "Question", "direction": "Linguistic direction"})
    st.subheader(":gray[Planned]")
    show(s["planned"], {"name": "Step", "setup": "Content", "direction": "Linguistic question"})
    st.info("**Reading the evidence.** " + s["note"])
    st.caption("The linguistic direction column states hypotheses suggested by the numbers, in the terms of "
               "`docs/LINGUISTICS.md`. The corpus establishes the patterns, not the accounts of them.")

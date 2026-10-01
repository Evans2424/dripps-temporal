import importlib.util
import re

import streamlit as st

from data import ROOT


def _viewer_helpers():
    """Reuse the static viewer's BibTeX reader rather than keeping a second copy."""
    spec = importlib.util.spec_from_file_location("viewer06", ROOT / "experiments/06_viewer.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def render():
    st.title("Methods and references")
    t1, t2, t3 = st.tabs(["Methods", "Linguistic background", "References"])
    with t1:
        md = (ROOT / "docs/methods.md").read_text(encoding="utf-8")
        st.markdown(re.sub(r"\[([a-z][\w\-]*(?:; ?[a-z][\w\-]*)*)\]", r"[`\1`]", md))
    with t2:
        st.markdown((ROOT / "docs/LINGUISTICS.md").read_text(encoding="utf-8"))
    with t3:
        refs = _viewer_helpers()._references((ROOT / "docs/references.bib").read_text(encoding="utf-8"))
        q = st.text_input("Search", placeholder="author, title or key")
        for r in refs:
            blob = " ".join(str(v) for v in r.values()).lower()
            if q and q.lower() not in blob:
                continue
            link = f"https://doi.org/{r['doi']}" if r["doi"] else r["url"]
            where = ", ".join(x for x in [r["venue"], r["volume"], r["pages"]] if x)
            st.markdown(f"`{r['key']}` · {r['author']} ({r['year']}). **{r['title']}**. {where}."
                        + (f" [link]({link})" if link else ""))

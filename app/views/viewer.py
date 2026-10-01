import importlib.util

import streamlit as st

from data import ROOT


@st.cache_resource(show_spinner="Building the viewer…")
def _html() -> str:
    """The standalone viewer, built from the tracked tables exactly as `make viewer` does."""
    spec = importlib.util.spec_from_file_location("viewer_build", ROOT / "experiments/06_viewer.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.render(mod.build_payload(), mod.TEMPLATE.read_text(encoding="utf-8"))


def render():
    st.title("Standalone viewer")
    st.iframe(_html(), height=1200)

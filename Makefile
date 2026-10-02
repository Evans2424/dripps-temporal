PY := .venv/bin/python

.PHONY: all ingest test audit baselines hierarchy varieties hierarchical explain viewer app clean

all: ingest test audit baselines hierarchy varieties hierarchical explain viewer

# converts the later-batch workbooks (Violeta, Abergaria) -> data/raw/dripps_{violeta,abergaria}.csv
ingest:
	$(PY) experiments/00_ingest.py

test:
	$(PY) -m pytest tests/ -q

audit:
	$(PY) experiments/01_audit.py

baselines:
	$(PY) experiments/02_baselines.py

hierarchy:
	$(PY) experiments/03_cue_hierarchy.py

varieties:
	$(PY) experiments/04_variety_hierarchies.py

# B6: pooled model, varying intercepts + slopes (slow: permutation tests, ~10 min)
hierarchical:
	$(PY) experiments/07_hierarchical.py

explain:
	$(PY) experiments/05_explain.py

# reads what the four steps above write, so it runs last
viewer:
	$(PY) experiments/06_viewer.py

# interactive explorer over the same tables; not part of `all` (it is a server)
app:
	.venv/bin/streamlit run app/streamlit_app.py

clean:
	rm -rf results/tables/* results/figures/* results/viewer.html .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

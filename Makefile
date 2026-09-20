PY := .venv/bin/python

.PHONY: all test audit baselines hierarchy varieties explain viewer clean

all: test audit baselines hierarchy varieties explain viewer

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

explain:
	$(PY) experiments/05_explain.py

# reads what the four steps above write, so it runs last
viewer:
	$(PY) experiments/06_viewer.py

clean:
	rm -rf results/tables/* results/figures/* results/viewer.html .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

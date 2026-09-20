PY := .venv/bin/python

.PHONY: all test audit baselines hierarchy varieties clean

all: test audit baselines hierarchy varieties

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

clean:
	rm -rf results/tables/* results/figures/* .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

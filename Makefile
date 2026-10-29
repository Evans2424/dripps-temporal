PY := .venv/bin/python

.PHONY: all ingest test audit baselines hierarchy varieties hierarchical encoder encoder-feat encoder-more decision explain viewer app clean

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

# T1/T3 encoder: needs a GPU + requirements-neural.txt, so not part of `all`
encoder:
	$(PY) experiments/08_encoder.py

encoder-feat:  # text + annotated cues; run after `make encoder`, tables then cover all four variants
	$(PY) experiments/08_encoder.py --variant plain_feat
	$(PY) experiments/08_encoder.py --variant marked_feat

encoder-more:  # residual / cues-as-text / auxiliary-cue variants, then plain + marked on BERTimbau, Albertina and mDeBERTa
	$(PY) experiments/08_encoder.py --variant new
	$(PY) experiments/08_encoder.py --variant both --checkpoint bertimbau
	$(PY) experiments/08_encoder.py --variant both --checkpoint albertina
	$(PY) experiments/08_encoder.py --variant both --checkpoint mdeberta

# T5 Jev-style decisions: one model per call (GPU, requirements-neural.txt), tables cover every model cached so far
decision:
	$(PY) experiments/10_decision.py --model qwen3.8-27b-fp8
	$(PY) experiments/10_decision.py --model qwen3.6-27b-fp8
	$(PY) experiments/10_decision.py --model qwen3.5-9b
	$(PY) experiments/10_decision.py --model qwen3.5-4b

explain:
	$(PY) experiments/05_explain.py

# reads what the four steps above write, so it runs last
viewer:
	$(PY) experiments/06_viewer.py

# interactive explorer over the same tables; not part of `all` (it is a server)
app:
	.venv/bin/streamlit run app/streamlit_app.py

clean:
	find results/tables -type f ! -name 't[135]_*' -delete  # t1_/t3_/t5_ need a GPU (make encoder, make decision); keep them
	rm -rf results/figures/* results/viewer.html .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

PY ?= .venv/bin/python
export HF_HUB_OFFLINE ?= 0

suite:      ## regenerate suites/geoint.yaml from the environment's oracles
	$(PY) scripts/make_geoint_suite.py
experiments: ## 4 models x prompts, reliability runs, repair runs (~2.5 h on an M4)
	./scripts/experiments.sh && ./scripts/followups.sh && ./scripts/v3.sh
grade:      ## deterministic + LLM-judge grading of every run
	$(PY) -m harness.cli grade runs/* --judge
validate-judge:
	$(PY) -m harness.cli validate-judge
report:
	$(PY) -m harness.cli report
gate:       ## the CI gate: current production prompt vs candidate (see gate.env)
	. ./gate.env && $(PY) -m harness.cli gate $$BASELINE $$CANDIDATE --mode $$MODE
test:
	$(PY) -m pytest -q

.PHONY: suite experiments grade validate-judge report gate test

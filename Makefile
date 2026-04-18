SHELL := /bin/bash
PY    := .venv/bin/python

# ----------------------------------------------------------------------------
# Server knobs. Override on the command line, e.g.
#     make train-deep CONFIG=configs/server.yaml CACHE=/mnt/scratch/dfp
# ----------------------------------------------------------------------------
CONFIG       ?= configs/default.yaml
SERVER_CFG   ?= configs/server.yaml
CACHE        ?= $(CURDIR)/.cache/deforest/patches
CHECKPOINTS  ?= $(CURDIR)/.cache/deforest/checkpoints
SUBMISSION   ?= submissions/submission.geojson
DEEP_CKPT    ?= $(CHECKPOINTS)/best.pt
GBM_MODEL    ?= models/gbm.txt
# Dataset root — symlinked to makeathon26/data/makeathon-challenge by default.
DATA_ROOT    ?= data/makeathon-challenge
MAKEATHON_DATA ?= ../makeathon26/data/makeathon-challenge
# PyTorch wheel index. ROCm on MI300X droplets; CPU on default Linux hosts.
TORCH_INDEX  ?= https://download.pytorch.org/whl/cpu
TORCH_PKGS   ?= torch torchvision

.PHONY: help install install-gpu install-torch link-data mock baseline train-gbm submit evaluate \
        preprocess train-deep submit-ensemble runtime test clean all

help:
	@echo "One-shot setup:"
	@echo "  make setup         Create .venv, install deps, link the dataset"
	@echo ""
	@echo "Local / CPU targets:"
	@echo "  install            Create .venv and install CPU requirements"
	@echo "  link-data          Symlink data/makeathon-challenge -> MAKEATHON_DATA"
	@echo "  mock               Generate a synthetic tile under data/makeathon-challenge/"
	@echo "  baseline           Produce a submission with the zero-training consensus model"
	@echo "  train-gbm          Train the LightGBM Tier-1 model"
	@echo "  submit             Produce a LightGBM submission (CPU-only)"
	@echo "  evaluate           Local metrics against training labels"
	@echo "  test               Run unit tests"
	@echo ""
	@echo "Server / GPU targets (CONFIG=configs/server.yaml):"
	@echo "  install-gpu        Install ROCm/CUDA PyTorch + GPU extras on top of install"
	@echo "  install-torch      Install CPU-only PyTorch (optional; enables deep targets)"
	@echo "  runtime            Print detected hardware + autoscaled defaults"
	@echo "  preprocess         Pre-compute per-tile feature caches"
	@echo "  train-deep         Train the ChangeUNet deep model"
	@echo "  submit-ensemble    Ensemble deep + GBM -> final submission.geojson"

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

setup: install link-data
	@echo "[setup] ready — try 'make baseline' or 'make test'"

.venv:
	./scripts/bootstrap_env.sh .venv

install: .venv
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip install -e .

install-gpu: install
	.venv/bin/pip install --index-url $(TORCH_INDEX) $(TORCH_PKGS)
	.venv/bin/pip install -r requirements-gpu.txt

# Installs the CPU-only PyTorch wheel so the `deep` targets become usable on
# a laptop / CI host without a GPU. Safe to run on top of `install`.
install-torch: install
	.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision
	.venv/bin/pip install -r requirements-gpu.txt

# Symlink the challenge dataset from ../makeathon26/data/makeathon-challenge
# into ./data/makeathon-challenge if it's not already present. This keeps
# the configured `data.root` working in the Docker workspace layout.
link-data:
	@if [[ -e $(DATA_ROOT) && ! -L $(DATA_ROOT) ]]; then \
	    echo "[link-data] $(DATA_ROOT) already exists (not a symlink) — leaving it alone"; \
	elif [[ -d $(MAKEATHON_DATA) ]]; then \
	    mkdir -p $$(dirname $(DATA_ROOT)); \
	    ln -sfn $(abspath $(MAKEATHON_DATA)) $(DATA_ROOT); \
	    echo "[link-data] $(DATA_ROOT) -> $$(readlink $(DATA_ROOT))"; \
	else \
	    echo "[link-data] $(MAKEATHON_DATA) not found — run 'make mock' to generate a synthetic tile"; \
	fi

# ---------------------------------------------------------------------------
# Laptop pipeline
# ---------------------------------------------------------------------------

mock: install
	$(PY) scripts/generate_mock_data.py --out $(DATA_ROOT) --tile MOCK_0_0

baseline: install
	$(PY) scripts/build_submission.py --model baseline --config $(CONFIG) \
	    --out submissions/baseline.geojson

train-gbm: install
	$(PY) scripts/train_gbm.py --config $(CONFIG) --out $(GBM_MODEL)

submit: install
	$(PY) scripts/build_submission.py --model gbm --config $(CONFIG) \
	    --gbm-model $(GBM_MODEL) --out submissions/gbm.geojson --split test

evaluate: install
	$(PY) scripts/evaluate.py --config $(CONFIG) \
	    --predictions submissions/baseline.geojson

# ---------------------------------------------------------------------------
# Server pipeline (MI300X / GPU)
# ---------------------------------------------------------------------------

runtime: install
	$(PY) -m deforest.cli runtime

preprocess: install
	$(PY) scripts/preprocess_tiles.py --config $(SERVER_CFG) --split train --cache-dir $(CACHE)
	$(PY) scripts/preprocess_tiles.py --config $(SERVER_CFG) --split test  --cache-dir $(CACHE)

train-deep: install
	$(PY) scripts/train_deep.py --config $(SERVER_CFG) \
	    --cache-dir $(CACHE) --checkpoint-dir $(CHECKPOINTS)

submit-ensemble: install
	$(PY) scripts/predict_ensemble.py --config $(SERVER_CFG) \
	    --deep-ckpt $(DEEP_CKPT) --gbm-model $(GBM_MODEL) \
	    --split test --cache-dir $(CACHE) --out $(SUBMISSION)

# ---------------------------------------------------------------------------

test: install
	.venv/bin/pytest

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -exec rm -rf {} +

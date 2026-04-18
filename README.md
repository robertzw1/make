# make — osapiens Makeathon 2026

Deforestation detection for the osapiens Makeathon 2026 challenge.

Produces a submission‑ready `FeatureCollection` GeoJSON where every polygon is
treated as deforestation after 2020.

## Quick start

```bash
# 1) install everything and symlink the challenge dataset
make setup

# 2) produce a zero-training consensus submission on the real data
make baseline

# 3) local metrics against the training labels (pseudo-GT = training consensus)
make evaluate
```

`make setup` is idempotent and does three things:

1. Builds `.venv/` (falls back to `pip.pyz` when `python3-venv` is missing,
   so it works on minimal Docker images).
2. `pip install -r requirements.txt && pip install -e .`
3. Symlinks `data/makeathon-challenge → ../makeathon26/data/makeathon-challenge`
   so the configured `data.root` resolves inside the workspace.

If the dataset lives elsewhere, override the source path:

```bash
make link-data MAKEATHON_DATA=/path/to/makeathon-challenge
```

If the dataset isn't available at all, generate a tiny synthetic tile
instead and run the pipeline on it:

```bash
make mock        # writes data/makeathon-challenge/{sentinel-1,2,...}/MOCK_0_0_*
make baseline    # uses the mock tile via configs/default.yaml
```

The output is written to `submissions/baseline.geojson` — upload this to the
leaderboard to verify the format round‑trips.

When the full dataset becomes available on your machine:

```bash
# Train the LightGBM pixel classifier on all training tiles.
.venv/bin/python scripts/train_gbm.py --out models/gbm.txt

# Then produce the leaderboard submission.
.venv/bin/python scripts/build_submission.py \
    --model gbm --gbm-model models/gbm.txt \
    --split test --out submissions/gbm.geojson
```

### macOS note — LightGBM needs libomp

```bash
brew install libomp
```

This is only required for Tier 1 (LightGBM training/prediction). The Tier 0
baseline and all tests run without it.

## Running on the AMD MI300X server

For the Ubuntu 24.04 droplet (1×MI300X / 192 GB HBM / 240 vCPU / 5 TB scratch)
use the dedicated config and `install-gpu` target:

```bash
make install-gpu TORCH_INDEX=https://download.pytorch.org/whl/rocm6.2
make runtime                           # print detected hardware + autoscaled defaults

# Pre-compute per-tile feature caches onto /mnt/scratch (240-way parallel).
make preprocess    SERVER_CFG=configs/server.yaml

# Train the deep ChangeUNet (bfloat16 AMP, batch_size=128 by default).
make train-deep    SERVER_CFG=configs/server.yaml

# Train LightGBM on the same features (240 CPU threads).
make train-gbm     CONFIG=configs/server.yaml

# Ensemble deep + GBM → final submission.
make submit-ensemble SERVER_CFG=configs/server.yaml
```

All server-specific settings live in [`configs/server.yaml`](configs/server.yaml);
see [`docs/ubuntu-mi300x.md`](docs/ubuntu-mi300x.md) for the full walk-through
(ROCm install, scratch-disk setup, VRAM/CPU budgeting, troubleshooting).

The autoscaling logic in [`src/deforest/runtime.py`](src/deforest/runtime.py)
detects ROCm/CUDA/CPU at run time, so the same code runs on a laptop, an NVIDIA
box, or the MI300X without any config changes — batch sizes, thread counts
and cache paths adjust themselves to the host.

## Why this approach

See [`docs/approach.md`](docs/approach.md) for the strategy, the analysis of
the ISPRS paper (Karaman et al. 2023, *BraDD‑S1TS* / U‑TAE), and the decision
to build on top of AlphaEarth Foundations + weak‑label fusion rather than a
pure SAR U‑TAE.

Short version: the paper's U‑TAE‑on‑Sentinel‑1 design is strong but
Brazil‑only, single‑modality, and untrained on weak labels — while the
challenge ships global AEF embeddings, three conflicting label sources, and
rewards polygon metrics including Year Accuracy. We treat U‑TAE as an
optional Tier 2 date‑refiner and make AEF + LightGBM the primary model.

## Layout

See [`docs/architecture.md`](docs/architecture.md).

## Generalization

All features are per‑pixel temporal differences on top of a globally
pretrained embedding (AEF) — the classifier never memorises region‑specific
reflectance statistics. Leave‑one‑region‑out validation is the recommended
protocol for estimating Africa / Indonesia performance from the training
tiles (mostly South America).

## Submission format recap

- GeoJSON FeatureCollection, `EPSG:4326`
- Each feature is `Polygon` or `MultiPolygon`
- Everything inside a polygon = deforestation
- Optional `properties.time_step` = `YYMM` (e.g. `2204`) or `null`

## License

MIT (see `LICENSE`).

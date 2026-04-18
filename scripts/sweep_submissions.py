#!/usr/bin/env python3
"""Generate several candidate submission GeoJSONs from a single inference pass.

We learned from the leaderboard that the GBM-only submission (324 polygons,
``time_step=None`` everywhere) scored 28.7%, while the 0.70/0.30 deep+gbm
ensemble (1737 polygons, ``time_step`` always set from the deep month head)
underperformed. The leading hypotheses:

* The polygon hygiene used for the ensemble (threshold 0.50, morph_open=2,
  min_area_ha=0.5) is too loose vs. GBM-only (threshold 0.55, morph_open=3).
* The deep month head collapsed to 7 distinct YYMM values in 2022; setting
  ``time_step=None`` (matching the canonical ``submission_utils.raster_to_geojson``
  default) is at minimum safe and probably better.
* The optimal ensemble weights are unknown; deep being the larger weight may
  actively hurt the leaderboard score.

This script runs deep + GBM inference once per cached test tile, caches the
two probability rasters in RAM, and then writes one GeoJSON per VARIANT
defined below. Submit whichever wins on the leaderboard.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rasterio import Affine
from rasterio.crs import CRS

from deforest.config import load_config
from deforest.deep.dataset import CachedTile
from deforest.deep.predict import (
    autoscale_infer,
    load_checkpoint,
    month_idx_to_yymm,
    predict_tile,
)
from deforest.ensemble import EnsembleWeights, blend
from deforest.models.gbm import PixelGBM
from deforest.postprocess.polygonize import (
    merge_feature_collections,
    polygonize,
    write_geojson,
)
from deforest.runtime import detect_hardware


@dataclass(frozen=True)
class Variant:
    """One submission recipe."""

    name: str
    out_path: str
    weight_deep: float
    weight_gbm: float
    threshold: float
    morph_open_px: int
    morph_close_px: int
    min_area_ha: float
    include_time_step: bool


# ---------------------------------------------------------------------------
# The actual sweep. Order matters only for log readability.
# ---------------------------------------------------------------------------
VARIANTS: list[Variant] = [
    # Sanity check: GBM-only through this pipeline. Should ~ match the
    # existing gbm.geojson (the leaderboard's 28.7% submission).
    Variant(
        name="gbm_only_sanity",
        out_path="submissions/sweep/gbm_only_sanity.geojson",
        weight_deep=0.0, weight_gbm=1.0,
        threshold=0.55, morph_open_px=3, morph_close_px=5,
        min_area_ha=0.5, include_time_step=False,
    ),
    # GBM-led ensemble: deep contributes a small re-ranking signal only.
    Variant(
        name="gbm_led_strict",
        out_path="submissions/sweep/gbm_led_strict.geojson",
        weight_deep=0.30, weight_gbm=0.70,
        threshold=0.55, morph_open_px=3, morph_close_px=5,
        min_area_ha=1.0, include_time_step=False,
    ),
    # 50/50 with strict post-processing.
    Variant(
        name="balanced_strict",
        out_path="submissions/sweep/balanced_strict.geojson",
        weight_deep=0.50, weight_gbm=0.50,
        threshold=0.55, morph_open_px=3, morph_close_px=5,
        min_area_ha=1.0, include_time_step=False,
    ),
    # Original weights but with the same strict polygon hygiene as GBM.
    Variant(
        name="deep_led_strict",
        out_path="submissions/sweep/deep_led_strict.geojson",
        weight_deep=0.70, weight_gbm=0.30,
        threshold=0.55, morph_open_px=3, morph_close_px=5,
        min_area_ha=1.0, include_time_step=False,
    ),
    # Deep-only with very strict post-processing.
    Variant(
        name="deep_only_strict",
        out_path="submissions/sweep/deep_only_strict.geojson",
        weight_deep=1.0, weight_gbm=0.0,
        threshold=0.60, morph_open_px=3, morph_close_px=5,
        min_area_ha=1.0, include_time_step=False,
    ),
    # Same as the current submission.geojson but with time_step removed.
    # Isolates the cost of the deep month head.
    Variant(
        name="current_no_ts",
        out_path="submissions/sweep/current_no_ts.geojson",
        weight_deep=0.70, weight_gbm=0.30,
        threshold=0.50, morph_open_px=2, morph_close_px=5,
        min_area_ha=0.5, include_time_step=False,
    ),
    # Strictly filtered subsets of the 28.7% file — cheapest paths to a
    # leaderboard improvement if the metric punishes small false positives.
    Variant(
        name="gbm_only_minarea1",
        out_path="submissions/sweep/gbm_only_minarea1.geojson",
        weight_deep=0.0, weight_gbm=1.0,
        threshold=0.55, morph_open_px=3, morph_close_px=5,
        min_area_ha=1.0, include_time_step=False,
    ),
    Variant(
        name="gbm_only_minarea2",
        out_path="submissions/sweep/gbm_only_minarea2.geojson",
        weight_deep=0.0, weight_gbm=1.0,
        threshold=0.55, morph_open_px=3, morph_close_px=5,
        min_area_ha=2.0, include_time_step=False,
    ),
    # Heaviest-GBM ensemble, very strict.
    Variant(
        name="gbm_heavy_extra_strict",
        out_path="submissions/sweep/gbm_heavy_extra_strict.geojson",
        weight_deep=0.20, weight_gbm=0.80,
        threshold=0.55, morph_open_px=3, morph_close_px=5,
        min_area_ha=2.0, include_time_step=False,
    ),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, default=Path("configs/server.yaml"))
    ap.add_argument("--deep-ckpt", type=Path,
                    default=Path(".cache/deforest/checkpoints/best.pt"))
    ap.add_argument("--gbm-model", type=Path, default=Path("models/gbm.txt"))
    ap.add_argument("--cache-dir", type=Path,
                    default=Path(".cache/deforest/patches"))
    ap.add_argument("--tiles", type=str, default=None,
                    help="Comma-separated tile ids (default: every tile in cache)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    hw = detect_hardware()
    print(f"[sweep] {hw.summary()}")

    deep_cfg = cfg.raw["deep"]
    cal_start = cfg.raw.get("month_calendar", {}).get("start", "2020-01")
    start_year, start_month = [int(v) for v in cal_start.split("-")[:2]]

    cache_dir = args.cache_dir
    if args.tiles:
        tile_ids = [t.strip() for t in args.tiles.split(",") if t.strip()]
    else:
        tile_ids = sorted(
            p.name for p in cache_dir.iterdir()
            if p.is_dir() and (p / "meta.json").exists()
        )
    if not tile_ids:
        print(f"[sweep] no tiles in {cache_dir}")
        return 1
    print(f"[sweep] tiles: {len(tile_ids)}")

    # --- load both models once ---
    device, amp_dtype, batch_size = autoscale_infer(deep_cfg)
    deep_model, _ = load_checkpoint(args.deep_ckpt, device=device)
    print(f"[sweep] deep loaded on {device} (bs={batch_size}, amp={amp_dtype})")
    gbm = PixelGBM().load(args.gbm_model)
    print(f"[sweep] gbm loaded from {args.gbm_model}")

    # Per-variant accumulators.
    fcs_per_variant: dict[str, list[dict]] = {v.name: [] for v in VARIANTS}

    # --- single inference pass ---
    for tid in tile_ids:
        print(f"[sweep] tile {tid}")
        tile = CachedTile.open(cache_dir / tid)
        transform = Affine(*tile.meta["transform"][:6])
        crs = CRS.from_string(tile.meta["crs"])
        forest = np.asarray(tile.forest, dtype=np.float32)

        # deep
        prob_deep, expected_idx = predict_tile(
            tile, deep_model,
            patch_size=int(deep_cfg.get("patch_size", 256)),
            overlap=int(deep_cfg.get("overlap", 64)),
            batch_size=batch_size,
            amp_dtype=amp_dtype,
            device=device,
        )
        prob_deep = (prob_deep * forest).astype(np.float32)
        month_yymm = month_idx_to_yymm(expected_idx, start_year, start_month).astype(np.int32)

        # gbm
        F, H, W = tile.features.shape
        X = np.asarray(tile.features, dtype=np.float32).reshape(F, H * W).T
        prob_gbm = gbm.predict_proba(X).reshape(H, W).astype(np.float32) * forest

        # polygonize per variant
        for v in VARIANTS:
            weights = EnsembleWeights(deep=v.weight_deep, gbm=v.weight_gbm)
            if v.weight_deep == 0.0:
                prob = prob_gbm
            elif v.weight_gbm == 0.0:
                prob = prob_deep
            else:
                prob = blend(prob_deep, prob_gbm, weights)

            fc = polygonize(
                prob,
                transform=transform,
                crs=crs,
                threshold=v.threshold,
                min_area_ha=v.min_area_ha,
                morph_open_px=v.morph_open_px,
                morph_close_px=v.morph_close_px,
                time_step_raster=month_yymm if v.include_time_step else None,
            )
            fcs_per_variant[v.name].append(fc)

    # --- write & summarise ---
    print("\n[sweep] summary")
    print(f"  {'variant':<22s} {'polygons':>10s}  {'size_MB':>8s}  {'path'}")
    for v in VARIANTS:
        merged = merge_feature_collections(fcs_per_variant[v.name])
        out = Path(v.out_path)
        write_geojson(merged, out)
        size_mb = out.stat().st_size / (1024 * 1024)
        print(f"  {v.name:<22s} {len(merged['features']):>10d}  "
              f"{size_mb:>7.2f}M  {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

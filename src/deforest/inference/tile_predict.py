"""Run any predictor tile-by-tile against the challenge dataset layout.

The main entry point is :func:`predict_tile`. It:

1. Picks the Sentinel-2 grid of the tile (CRS, transform, shape) — all other
   rasters get reprojected onto it.
2. Loads and aligns AEF embeddings for the relevant years.
3. Loads weak labels (train split only — test tiles won't have any).
4. Computes S1 / S2 annual statistics.
5. Builds the 2020 forest mask.
6. Fuses weak labels (for training weights and for Tier-0).
7. Runs either the baseline or the LightGBM model.
8. Returns ``(prob, time_step, crs, transform)``.

Callers are expected to post-process ``prob`` into polygons separately via
``postprocess.polygonize``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.warp import Resampling

from ..data.align import reproject_multiband_to_grid, reproject_to_grid
from ..data.forest_mask import forest_mask_from_aef, forest_mask_from_ndvi
from ..data.paths import DataPaths
from ..data.readers import list_s1_months, list_s2_months, read_aef, read_s2
from ..features.aef import aef_features
from ..features.satellite import pack_features, s1_annual_stats, s2_annual_stats
from ..labels.fusion import fuse, FusedLabels
from ..labels.parsers import WeakLabel, parse_gladl, parse_glads2, parse_radd
from ..models.baseline import baseline_predict
from ..models.gbm import PixelGBM


@dataclass
class TilePrediction:
    tile_id: str
    prob: np.ndarray       # (H, W) float32
    time_step: np.ndarray  # (H, W) int32, YYMM or 0
    crs: CRS
    transform: object
    fused: FusedLabels | None = None  # kept for evaluation


def predict_tile(
    tile_id: str,
    paths: DataPaths,
    *,
    split: str = "train",
    model: Literal["baseline", "gbm"] = "baseline",
    gbm: PixelGBM | None = None,
    fusion_cfg: dict | None = None,
    year_base: int = 2020,
    year_last: int | None = None,
) -> TilePrediction:
    # ---- 1. Pick the Sentinel-2 reference grid ---------------------------
    s2_dir = paths.s2_dir(tile_id, split=split)
    s2_entries = list_s2_months(s2_dir)
    if not s2_entries:
        raise FileNotFoundError(f"No Sentinel-2 data for tile {tile_id} under {s2_dir}")

    ref_year, ref_month = s2_entries[0]
    ref_tif = paths.s2_tif(tile_id, ref_year, ref_month, split=split)
    with rasterio.open(ref_tif) as src:
        ref_crs = src.crs
        ref_transform = src.transform
        ref_shape = (src.height, src.width)

    # ---- 2. AEF per year, reprojected to the S2 grid ---------------------
    years_available = sorted({y for (y, _) in s2_entries})
    if year_base not in years_available:
        year_base = years_available[0]
    if year_last is None:
        year_last = years_available[-1]

    aef_by_year: dict[int, np.ndarray] = {}
    for y in (year_base, year_last):
        p = paths.aef_tiff(tile_id, y, split=split)
        if not p.exists():
            continue
        data, profile = read_aef(p)
        aef_by_year[y] = reproject_multiband_to_grid(
            data,
            src_transform=profile["transform"],
            src_crs=profile["crs"],
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            dst_shape=ref_shape,
            resampling=Resampling.bilinear,
        )

    if not aef_by_year:
        raise FileNotFoundError(
            f"No AEF years available for tile {tile_id}; expected {year_base} and/or {year_last}"
        )
    aef_feats = aef_features(aef_by_year)

    # ---- 3. S1/S2 annual statistics --------------------------------------
    s1_dir = paths.s1_dir(tile_id, split=split)
    grid = dict(ref_transform=ref_transform, ref_crs=ref_crs, ref_shape=ref_shape)
    s1_base = s1_annual_stats(s1_dir, year_base, **grid)
    s1_last = s1_annual_stats(s1_dir, year_last, **grid)
    s2_base = s2_annual_stats(s2_dir, year_base, **grid)
    s2_last = s2_annual_stats(s2_dir, year_last, **grid)

    # ---- 4. Weak labels (train split only) -------------------------------
    fused: FusedLabels | None = None
    if split == "train":
        fused = _load_and_fuse_labels(
            tile_id,
            paths,
            ref_crs=ref_crs,
            ref_transform=ref_transform,
            ref_shape=ref_shape,
            fusion_cfg=fusion_cfg or {},
        )

    # ---- 5. Forest mask (2020) -------------------------------------------
    forest_mask = _build_forest_mask_2020(aef_by_year, year_base, s2_base)

    # ---- 6. Run requested model ------------------------------------------
    if model == "baseline":
        if fused is None:
            # Test-time: no labels → produce an empty prediction.
            prob = np.zeros(ref_shape, dtype=np.float32)
            time_step = np.zeros(ref_shape, dtype=np.int32)
        else:
            prob, time_step = baseline_predict(fused)
        prob *= forest_mask.astype(np.float32)
        time_step *= forest_mask.astype(np.int32)
        return TilePrediction(tile_id, prob, time_step, ref_crs, ref_transform, fused)

    if model == "gbm":
        if gbm is None:
            raise ValueError("model='gbm' requires a PixelGBM instance")
        X, _names = pack_features(aef_feats, s2_base, s2_last, s1_base, s1_last)
        prob_flat = gbm.predict_proba(X)
        prob = prob_flat.reshape(ref_shape).astype(np.float32)
        prob *= forest_mask.astype(np.float32)
        # time_step: inherit from weak labels when available, otherwise 0.
        if fused is not None:
            _, time_step = baseline_predict(fused)
        else:
            time_step = np.zeros(ref_shape, dtype=np.int32)
        return TilePrediction(tile_id, prob, time_step, ref_crs, ref_transform, fused)

    raise ValueError(f"Unknown model '{model}'")


# ---------------------------------------------------------------------------


def _load_and_fuse_labels(
    tile_id: str,
    paths: DataPaths,
    *,
    ref_crs,
    ref_transform,
    ref_shape: tuple[int, int],
    fusion_cfg: dict,
) -> FusedLabels | None:
    sources: dict[str, WeakLabel] = {}

    # RADD
    radd_path = paths.radd(tile_id)
    if radd_path.exists():
        with rasterio.open(radd_path) as src:
            radd_raw = src.read(1)
            radd_aligned = reproject_to_grid(
                radd_raw,
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=ref_transform, dst_crs=ref_crs,
                dst_shape=ref_shape, resampling=Resampling.nearest,
                src_nodata=0, dst_nodata=0, dtype=np.int32,
            )
        sources["radd"] = parse_radd(radd_aligned)

    # GLAD-L: combine all available years
    gladl_combined: WeakLabel | None = None
    for yy in range(20, 30):
        alert_p = paths.gladl_alert(tile_id, yy)
        date_p = paths.gladl_date(tile_id, yy)
        if not (alert_p.exists() and date_p.exists()):
            continue
        with rasterio.open(alert_p) as src_a, rasterio.open(date_p) as src_d:
            alert_raw = src_a.read(1)
            date_raw = src_d.read(1)
            alert_aligned = reproject_to_grid(
                alert_raw, src_transform=src_a.transform, src_crs=src_a.crs,
                dst_transform=ref_transform, dst_crs=ref_crs,
                dst_shape=ref_shape, resampling=Resampling.nearest,
                src_nodata=0, dst_nodata=0, dtype=np.uint8,
            )
            date_aligned = reproject_to_grid(
                date_raw, src_transform=src_d.transform, src_crs=src_d.crs,
                dst_transform=ref_transform, dst_crs=ref_crs,
                dst_shape=ref_shape, resampling=Resampling.nearest,
                src_nodata=0, dst_nodata=0, dtype=np.uint16,
            )
        wl = parse_gladl(alert_aligned, date_aligned, yy=yy)
        if gladl_combined is None:
            gladl_combined = wl
        else:
            mask = wl.confidence > gladl_combined.confidence
            gladl_combined.confidence[mask] = wl.confidence[mask]
            gladl_combined.days[mask] = wl.days[mask]
    if gladl_combined is not None:
        sources["gladl"] = gladl_combined

    # GLAD-S2
    a_p = paths.glads2_alert(tile_id)
    d_p = paths.glads2_date(tile_id)
    if a_p.exists() and d_p.exists():
        with rasterio.open(a_p) as src_a, rasterio.open(d_p) as src_d:
            alert_raw = src_a.read(1)
            date_raw = src_d.read(1)
            alert_aligned = reproject_to_grid(
                alert_raw, src_transform=src_a.transform, src_crs=src_a.crs,
                dst_transform=ref_transform, dst_crs=ref_crs,
                dst_shape=ref_shape, resampling=Resampling.nearest,
                src_nodata=0, dst_nodata=0, dtype=np.uint8,
            )
            date_aligned = reproject_to_grid(
                date_raw, src_transform=src_d.transform, src_crs=src_d.crs,
                dst_transform=ref_transform, dst_crs=ref_crs,
                dst_shape=ref_shape, resampling=Resampling.nearest,
                src_nodata=0, dst_nodata=0, dtype=np.uint16,
            )
        sources["glads2"] = parse_glads2(alert_aligned, date_aligned)

    if not sources:
        return None

    return fuse(
        sources,
        agreement_threshold=fusion_cfg.get("agreement_threshold", 0.7),
        single_threshold=fusion_cfg.get("single_threshold", 0.9),
    )


def _build_forest_mask_2020(
    aef_by_year: dict[int, np.ndarray],
    year_base: int,
    s2_base: dict[str, np.ndarray] | None,
) -> np.ndarray:
    """AEF-clustering with NDVI as the disambiguator; fallback to NDVI threshold."""
    ndvi_med = None if s2_base is None else s2_base.get("median_ndvi")
    if year_base in aef_by_year:
        try:
            return forest_mask_from_aef(aef_by_year[year_base], ndvi_2020_median=ndvi_med)
        except Exception:  # pragma: no cover - defensive
            pass
    if ndvi_med is not None:
        return forest_mask_from_ndvi(ndvi_med)
    # No data → everything is forest by default (least-harm).
    any_year = next(iter(aef_by_year.values()))
    return np.ones(any_year.shape[1:], dtype=bool)

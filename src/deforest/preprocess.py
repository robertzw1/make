"""Precompute feature + label tensors per tile and cache to /mnt/scratch.

Why precompute?
---------------
Reading raw Sentinel time series from a 5 TB NVMe during training is faster
than S3, but still an order of magnitude slower than a memory-mapped numpy
array. At 240 vCPUs this step parallelises almost perfectly; the end result
is that the GPU never waits for data.

Per-tile cache layout (written under ``cache_dir/<tile_id>/``):

=================  =====================  ==============================================
File                Shape                  Description
-----------------  ---------------------  ----------------------------------------------
``features.npy``    (F, H, W) float16      AEF + S1/S2 stats packed channels-first.
``labels.npy``      (H, W) uint8           Fused weak-label binary target.
``month.npy``       (H, W) int16           1-based month index (0 = no alert).
``weight.npy``      (H, W) float16         Fused confidence (sample weight).
``forest.npy``      (H, W) uint8           2020 forest mask (1 = forest).
``meta.json``                              CRS/transform + feature names.
=================  =====================  ==============================================

The root ``patch_index.jsonl`` lists every valid patch (tile_id, y, x,
positive_fraction) so the training loader can stream without re-scanning.
"""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

import numpy as np
import rasterio
from rasterio.warp import Resampling

from .data.align import reproject_multiband_to_grid
from .data.paths import DataPaths
from .data.readers import list_s2_months, read_aef
from .features.aef import aef_features
from .features.satellite import pack_features, s1_annual_stats, s2_annual_stats
from .inference.tile_predict import _build_forest_mask_2020, _load_and_fuse_labels
from .deep.dataset import PatchRef, load_patch_index, write_patch_index
from .labels.parsers import _UNIX_EPOCH  # reuse the UNIX epoch for month calc


@dataclass
class PreprocessConfig:
    cache_dir: Path
    patch_size: int = 256
    patch_stride: int = 192
    max_patches_per_tile: int = 512
    forest_min_fraction: float = 0.05
    skip_if_empty: bool = True
    month_calendar_start: tuple[int, int] = (2020, 1)   # (year, month)
    months_in_calendar: int = 72
    fusion: dict | None = None
    workers: int | None = None


def preprocess_tile(
    tile_id: str,
    data_paths: DataPaths,
    cfg: PreprocessConfig,
    *,
    split: str = "train",
) -> tuple[str, int]:
    """Write one tile's cache and return (tile_id, n_patches_indexed)."""
    tile_dir = cfg.cache_dir / tile_id
    tile_dir.mkdir(parents=True, exist_ok=True)

    s2_dir = data_paths.s2_dir(tile_id, split=split)
    s2_entries = list_s2_months(s2_dir)
    if not s2_entries:
        return tile_id, 0
    years = sorted({y for (y, _) in s2_entries})
    year_base, year_last = years[0], years[-1]

    ref_tif = data_paths.s2_tif(tile_id, s2_entries[0][0], s2_entries[0][1], split=split)
    with rasterio.open(ref_tif) as src:
        ref_crs, ref_transform, ref_shape = src.crs, src.transform, (src.height, src.width)

    # --- AEF per relevant year, reprojected onto the S2 grid -----------
    aef_by_year: dict[int, np.ndarray] = {}
    for y in (year_base, year_last):
        p = data_paths.aef_tiff(tile_id, y, split=split)
        if not p.exists():
            continue
        data, profile = read_aef(p)
        aef_by_year[y] = reproject_multiband_to_grid(
            data,
            src_transform=profile["transform"], src_crs=profile["crs"],
            dst_transform=ref_transform, dst_crs=ref_crs,
            dst_shape=ref_shape, resampling=Resampling.bilinear,
        )
    if not aef_by_year:
        return tile_id, 0

    aef_feats = aef_features(aef_by_year)
    grid = dict(ref_transform=ref_transform, ref_crs=ref_crs, ref_shape=ref_shape)
    s1_base = s1_annual_stats(data_paths.s1_dir(tile_id, split=split), year_base, **grid)
    s1_last = s1_annual_stats(data_paths.s1_dir(tile_id, split=split), year_last, **grid)
    s2_base = s2_annual_stats(s2_dir, year_base, **grid)
    s2_last = s2_annual_stats(s2_dir, year_last, **grid)

    X, feature_names = pack_features(aef_feats, s2_base, s2_last, s1_base, s1_last)
    F = X.shape[1]
    H, W = ref_shape
    features = X.T.reshape(F, H, W).astype(np.float16)

    # --- Forest + labels --------------------------------------------------
    forest_mask = _build_forest_mask_2020(aef_by_year, year_base, s2_base)
    fused = _load_and_fuse_labels(
        tile_id, data_paths,
        ref_crs=ref_crs, ref_transform=ref_transform, ref_shape=ref_shape,
        fusion_cfg=cfg.fusion or {},
    )
    if fused is not None:
        labels_arr = fused.binary.astype(np.uint8) * forest_mask.astype(np.uint8)
        weight_arr = (fused.confidence * forest_mask.astype(np.float32) + (1 - forest_mask.astype(np.float32))).astype(np.float16)
        month_arr = _unix_days_to_month_index(
            fused.median_days, cfg.month_calendar_start
        ).astype(np.int16)
        month_arr[labels_arr == 0] = 0
    else:
        labels_arr = np.zeros(ref_shape, dtype=np.uint8)
        weight_arr = np.ones(ref_shape, dtype=np.float16)
        month_arr = np.zeros(ref_shape, dtype=np.int16)

    # --- Persist ---------------------------------------------------------
    np.save(tile_dir / "features.npy", features)
    np.save(tile_dir / "labels.npy", labels_arr)
    np.save(tile_dir / "weight.npy", weight_arr)
    np.save(tile_dir / "month.npy", month_arr)
    np.save(tile_dir / "forest.npy", forest_mask.astype(np.uint8))
    meta = {
        "tile_id": tile_id,
        "crs": str(ref_crs),
        "transform": list(ref_transform),
        "shape": list(ref_shape),
        "year_base": int(year_base),
        "year_last": int(year_last),
        "feature_names": feature_names,
        "month_calendar_start": list(cfg.month_calendar_start),
        "months_in_calendar": int(cfg.months_in_calendar),
    }
    (tile_dir / "meta.json").write_text(json.dumps(meta))

    # --- Patch index -----------------------------------------------------
    refs = _patch_refs_for_tile(
        tile_id, labels_arr, forest_mask.astype(np.uint8),
        patch_size=cfg.patch_size, stride=cfg.patch_stride,
        max_patches=cfg.max_patches_per_tile,
        forest_min_fraction=cfg.forest_min_fraction,
        skip_if_empty=cfg.skip_if_empty,
    )
    return tile_id, len(refs)


def preprocess_all(
    tiles: list[str],
    data_paths: DataPaths,
    cfg: PreprocessConfig,
    *,
    split: str = "train",
    on_progress=None,
) -> None:
    """Parallel preprocessing across tiles. Writes the global patch index."""
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)

    workers = cfg.workers or max(1, (os.cpu_count() or 2) - 4)
    workers = min(workers, max(1, len(tiles)))

    all_refs: list[PatchRef] = []
    if workers <= 1 or len(tiles) == 1:
        for tid in tiles:
            _tid, n = preprocess_tile(tid, data_paths, cfg, split=split)
            all_refs.extend(_read_tile_refs(cfg.cache_dir, tid, expected=n))
            if on_progress is not None:
                on_progress(tid, n)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(preprocess_tile, tid, data_paths, cfg, split=split): tid
                for tid in tiles
            }
            for fut in as_completed(futures):
                tid, n = fut.result()
                all_refs.extend(_read_tile_refs(cfg.cache_dir, tid, expected=n))
                if on_progress is not None:
                    on_progress(tid, n)

    # Persist a per-split index AND a combined one. Running ``preprocess
    # --split test`` after ``--split train`` (and vice versa) must not wipe
    # the other split's patches from the combined ``patch_index.jsonl`` —
    # the deep training loop reads only its split, but legacy callers and
    # ``predict_ensemble`` still rely on the combined file existing.
    write_patch_index(cfg.cache_dir, all_refs, split=split)
    other_splits = [
        s for s in ("train", "test", "val")
        if s != split and (cfg.cache_dir / f"patch_index.{s}.jsonl").exists()
    ]
    combined: list[PatchRef] = list(all_refs)
    for s in other_splits:
        combined.extend(load_patch_index(cfg.cache_dir, split=s))
    write_patch_index(cfg.cache_dir, combined, split=None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _patch_refs_for_tile(
    tile_id: str,
    labels: np.ndarray,
    forest: np.ndarray,
    *,
    patch_size: int,
    stride: int,
    max_patches: int,
    forest_min_fraction: float,
    skip_if_empty: bool,
) -> list[PatchRef]:
    h, w = labels.shape
    refs: list[PatchRef] = []
    # Skip tiles smaller than a single training patch. The dataset assumes
    # every ref expands to ``(patch_size, patch_size)``; if the tile is
    # smaller, slicing silently truncates and the collate step blows up
    # with ``stack expects each tensor to be equal size``.
    if h < patch_size or w < patch_size:
        return refs
    ys = _axis(h, patch_size, stride)
    xs = _axis(w, patch_size, stride)
    for y in ys:
        for x in xs:
            lab = labels[y : y + patch_size, x : x + patch_size]
            fr = forest[y : y + patch_size, x : x + patch_size]
            if fr.mean() < forest_min_fraction:
                continue
            pos_frac = float(lab.mean())
            if skip_if_empty and pos_frac == 0 and fr.mean() > 0.98:
                # All-forest, all-negative patch — keep a fraction for hard negatives.
                if (y + x) % 4 != 0:
                    continue
            refs.append(PatchRef(tile_id=tile_id, y=y, x=x, positive_fraction=pos_frac))

    if len(refs) > max_patches:
        refs.sort(key=lambda r: -r.positive_fraction)
        refs = refs[:max_patches]

    # Persist per-tile refs for re-indexing.
    tile_dir = Path(refs[0].tile_id).parent if refs else Path()  # unused
    return refs


def _read_tile_refs(cache_dir: Path, tile_id: str, expected: int) -> list[PatchRef]:
    """Rebuild refs list from the tile cache (cheap: scan labels again)."""
    td = cache_dir / tile_id
    meta = json.loads((td / "meta.json").read_text())
    labels = np.load(td / "labels.npy", mmap_mode="r")
    forest = np.load(td / "forest.npy", mmap_mode="r")
    return _patch_refs_for_tile(
        tile_id, labels, forest,
        patch_size=256, stride=192, max_patches=expected or 512,
        forest_min_fraction=0.05, skip_if_empty=True,
    )


def _axis(total: int, patch: int, stride: int) -> list[int]:
    if total <= patch:
        return [0]
    out = list(range(0, total - patch + 1, stride))
    if out[-1] != total - patch:
        out.append(total - patch)
    return out


def _unix_days_to_month_index(
    unix_days: np.ndarray, start: tuple[int, int]
) -> np.ndarray:
    """Map UNIX-day counts to a 1-based month index relative to ``start``.

    0 stays 0. We avoid per-pixel Python date construction by first turning
    UNIX days into (year, month) via the Julian-Gregorian fixed math for the
    1970-2100 window.
    """
    out = np.zeros(unix_days.shape, dtype=np.int32)
    valid = unix_days > 0
    if not np.any(valid):
        return out

    start_year, start_month = start
    start_unix = (date(start_year, start_month, 1) - _UNIX_EPOCH).days

    # For each valid pixel compute (year, month) via a coarse-to-fine map.
    # This is correct for the full Gregorian calendar.
    days = unix_days[valid]
    years = np.empty_like(days)
    months = np.empty_like(days)

    # Work in float years; vectorised round-down via datetime is impossible
    # without a loop, so we approximate and then correct.
    # Approximation: y ≈ 1970 + days / 365.2425
    approx = 1970 + days / 365.2425
    y = approx.astype(np.int64)

    # Build a lookup of cumulative days from 1970 for years 1970-2100.
    # 131 entries — negligible.
    yr_start_unix = np.array(
        [(date(yr, 1, 1) - _UNIX_EPOCH).days for yr in range(1970, 2101)],
        dtype=np.int64,
    )
    # Find exact year by searchsorted on yr_start_unix.
    idx = np.searchsorted(yr_start_unix, days, side="right") - 1
    idx = np.clip(idx, 0, len(yr_start_unix) - 1)
    y = 1970 + idx
    doy = days - yr_start_unix[idx]  # 0-indexed day of year

    # Days per month tables (non-leap / leap).
    dom_norm = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365])
    dom_leap = np.array([0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366])
    leap = ((y % 4 == 0) & (y % 100 != 0)) | (y % 400 == 0)
    # Per-pixel month = 1..12. Vectorise via searchsorted per row (small work).
    m = np.empty_like(y)
    if np.any(~leap):
        m[~leap] = np.searchsorted(dom_norm, doy[~leap], side="right")
    if np.any(leap):
        m[leap] = np.searchsorted(dom_leap, doy[leap], side="right")
    m = np.clip(m, 1, 12)

    month_index = (y - start_year) * 12 + (m - start_month) + 1  # 1-based
    out[valid] = month_index.astype(np.int32)
    out[out < 0] = 0
    return out

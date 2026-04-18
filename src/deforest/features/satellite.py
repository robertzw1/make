"""Annual Sentinel-1 and Sentinel-2 statistics.

We deliberately avoid full-time-series features at this stage — they would be
expensive and the foundation-model AEF embeddings already summarise them.
Instead we compute *annual* statistics (per pixel) for 2020 and the most
recent available year, plus their deltas, yielding ~15 scalar features per
pixel.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling

from ..data.align import reproject_multiband_to_grid, reproject_to_grid
from ..data.readers import read_s1, read_s2, list_s1_months, list_s2_months


# --- NDVI / NDMI helpers ---------------------------------------------------


def ndvi_from_s2(bands: np.ndarray) -> np.ndarray:
    """NDVI = (NIR - RED) / (NIR + RED) using B08 (index 7) and B04 (index 3)."""
    nir = bands[7].astype(np.float32)
    red = bands[3].astype(np.float32)
    denom = nir + red
    ndvi = np.where(denom > 0, (nir - red) / np.maximum(denom, 1e-6), 0.0)
    return ndvi.astype(np.float32)


def ndmi_from_s2(bands: np.ndarray) -> np.ndarray:
    """NDMI = (NIR - SWIR) / (NIR + SWIR) using B08 and B11 (index 10)."""
    nir = bands[7].astype(np.float32)
    swir = bands[10].astype(np.float32)
    denom = nir + swir
    ndmi = np.where(denom > 0, (nir - swir) / np.maximum(denom, 1e-6), 0.0)
    return ndmi.astype(np.float32)


# --- Annual statistics -----------------------------------------------------


def s2_annual_stats(
    s2_dir: str | Path,
    year: int,
    *,
    ref_transform=None,
    ref_crs=None,
    ref_shape: tuple[int, int] | None = None,
) -> dict[str, np.ndarray] | None:
    """Compute median NDVI/NDMI and NDVI slope for the given year.

    Real-world tiles often have monthly composites with slightly different
    extents/transforms. If ``ref_*`` are provided each monthly composite is
    resampled onto that grid before stacking; otherwise the first composite
    we successfully read defines the reference (and later mismatched
    composites get reprojected onto it).
    """
    months = [m for (y, m) in list_s2_months(s2_dir) if y == year]
    if not months:
        return None

    ndvi_stack, ndmi_stack = [], []
    for m in months:
        tif = Path(s2_dir) / f"{Path(s2_dir).name}_{year}_{m}.tif"
        if not tif.exists():
            continue
        bands, prof = read_s2(tif)
        if ref_transform is None or ref_crs is None or ref_shape is None:
            # Use this composite as the implicit reference for the rest.
            ref_transform = prof["transform"]
            ref_crs = prof["crs"]
            ref_shape = prof["shape"]
        elif bands.shape[1:] != ref_shape or prof["transform"] != ref_transform:
            bands = reproject_multiband_to_grid(
                bands,
                src_transform=prof["transform"], src_crs=prof["crs"],
                dst_transform=ref_transform, dst_crs=ref_crs,
                dst_shape=ref_shape, resampling=Resampling.bilinear,
            )
        ndvi_stack.append(ndvi_from_s2(bands))
        ndmi_stack.append(ndmi_from_s2(bands))

    if not ndvi_stack:
        return None
    ndvi = np.stack(ndvi_stack, axis=0)
    ndmi = np.stack(ndmi_stack, axis=0)

    median_ndvi = np.nanmedian(ndvi, axis=0).astype(np.float32)
    median_ndmi = np.nanmedian(ndmi, axis=0).astype(np.float32)
    min_ndvi = np.nanmin(ndvi, axis=0).astype(np.float32)

    # NDVI slope across months (least-squares on the month index).
    t = np.arange(ndvi.shape[0], dtype=np.float32)
    if ndvi.shape[0] >= 2:
        t_mean = t.mean()
        v_mean = np.nanmean(ndvi, axis=0)
        num = np.nansum((t - t_mean)[:, None, None] * (ndvi - v_mean[None]), axis=0)
        den = np.nansum((t - t_mean) ** 2)
        slope_ndvi = (num / max(den, 1e-6)).astype(np.float32)
    else:
        slope_ndvi = np.zeros_like(median_ndvi)

    return {
        "median_ndvi": median_ndvi,
        "median_ndmi": median_ndmi,
        "min_ndvi": min_ndvi,
        "slope_ndvi": slope_ndvi,
    }


def s1_annual_stats(
    s1_dir: str | Path,
    year: int,
    *,
    ref_transform=None,
    ref_crs=None,
    ref_shape: tuple[int, int] | None = None,
) -> dict[str, np.ndarray] | None:
    """Mean / std / min of VV in dB for the given year (any orbit).

    When a reference grid is supplied each monthly Sentinel-1 raster is
    resampled onto it (S1 ships at 30 m, S2 at 10 m, so callers will
    typically pass the S2 grid as the reference).
    """
    entries = [(y, m, o) for (y, m, o) in list_s1_months(s1_dir) if y == year]
    if not entries:
        return None

    stack = []
    for (y, m, o) in entries:
        tif = Path(s1_dir) / f"{Path(s1_dir).name}_{y}_{m}_{o}.tif"
        if not tif.exists():
            continue
        vv, prof = read_s1(tif)
        if ref_transform is not None and ref_crs is not None and ref_shape is not None:
            if vv.shape != ref_shape or prof["transform"] != ref_transform:
                vv = reproject_to_grid(
                    vv,
                    src_transform=prof["transform"], src_crs=prof["crs"],
                    dst_transform=ref_transform, dst_crs=ref_crs,
                    dst_shape=ref_shape, resampling=Resampling.bilinear,
                    src_nodata=np.nan, dst_nodata=np.nan, dtype=np.float32,
                )
        stack.append(vv)
    if not stack:
        return None
    # Allow any single-month-only mismatch by aligning all S1 frames to the
    # first one we read when no explicit reference grid is given.
    if not all(a.shape == stack[0].shape for a in stack):
        target = stack[0].shape
        stack = [a if a.shape == target else _crop_or_pad(a, target) for a in stack]
    arr = np.stack(stack, axis=0)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return {
                "mean_vv": np.nanmean(arr, axis=0).astype(np.float32),
                "std_vv": np.nanstd(arr, axis=0).astype(np.float32),
                "min_vv": np.nanmin(arr, axis=0).astype(np.float32),
            }


def _crop_or_pad(arr: np.ndarray, target: tuple[int, int]) -> np.ndarray:
    """Centre-crop or zero-pad a 2-D array to ``target`` shape (rare fallback)."""
    h, w = arr.shape
    th, tw = target
    out = np.full(target, np.nan if np.issubdtype(arr.dtype, np.floating) else 0, dtype=arr.dtype)
    sh = min(h, th); sw = min(w, tw)
    out[:sh, :sw] = arr[:sh, :sw]
    return out


def pack_features(
    aef_feats: dict[str, np.ndarray],
    s2_base: dict[str, np.ndarray] | None,
    s2_last: dict[str, np.ndarray] | None,
    s1_base: dict[str, np.ndarray] | None,
    s1_last: dict[str, np.ndarray] | None,
) -> tuple[np.ndarray, list[str]]:
    """Stack all features into (H*W, F). Returns (X, feature_names)."""
    layers: list[np.ndarray] = []
    names: list[str] = []

    base = aef_feats["aef_base"]
    last = aef_feats["aef_last"]
    delta = aef_feats["aef_delta"]
    h, w = base.shape[1:]
    n = h * w

    layers.append(base.reshape(base.shape[0], -1).T)
    names += [f"aef_base_{i}" for i in range(base.shape[0])]
    layers.append(last.reshape(last.shape[0], -1).T)
    names += [f"aef_last_{i}" for i in range(last.shape[0])]
    layers.append(delta.reshape(delta.shape[0], -1).T)
    names += [f"aef_delta_{i}" for i in range(delta.shape[0])]
    layers.append(aef_feats["aef_norm"].reshape(-1, 1))
    names.append("aef_norm")

    def add_scalar(m: dict[str, np.ndarray] | None, keys: list[str], prefix: str) -> None:
        for k in keys:
            v = None if m is None else m.get(k)
            if v is None:
                v = np.zeros((h, w), dtype=np.float32)
            layers.append(v.reshape(-1, 1))
            names.append(f"{prefix}_{k}")

    add_scalar(s1_base, ["mean_vv", "std_vv", "min_vv"], "s1_2020")
    add_scalar(s1_last, ["mean_vv", "std_vv", "min_vv"], "s1_last")
    add_scalar(s2_base, ["median_ndvi", "median_ndmi", "min_ndvi", "slope_ndvi"], "s2_2020")
    add_scalar(s2_last, ["median_ndvi", "median_ndmi", "min_ndvi", "slope_ndvi"], "s2_last")

    # Deltas
    for stats_base, stats_last, keys, prefix in [
        (s1_base, s1_last, ["mean_vv", "std_vv"], "s1_delta"),
        (s2_base, s2_last, ["median_ndvi", "median_ndmi"], "s2_delta"),
    ]:
        for k in keys:
            vb = (stats_base or {}).get(k, np.zeros((h, w), dtype=np.float32))
            vl = (stats_last or {}).get(k, np.zeros((h, w), dtype=np.float32))
            layers.append((vl - vb).reshape(-1, 1))
            names.append(f"{prefix}_{k}")

    X = np.concatenate(layers, axis=1)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    assert X.shape == (n, len(names)), f"{X.shape} vs {len(names)}"
    return X, names

"""Train the LightGBM pixel-classifier on the training split.

Iterates over training tiles, builds features + fused weak labels, subsamples
pixels per-tile, concatenates, trains, and writes the booster to disk.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from deforest.config import load_config
from deforest.cli import _data_paths, _resolve_tiles
from deforest.features.aef import aef_features
from deforest.features.satellite import pack_features, s1_annual_stats, s2_annual_stats
from deforest.inference.tile_predict import _build_forest_mask_2020, _load_and_fuse_labels
from deforest.data.align import reproject_multiband_to_grid
from deforest.data.readers import read_aef, list_s2_months
from deforest.models.gbm import GBMConfig, PixelGBM, subsample_pixels

import rasterio
from rasterio.warp import Resampling


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--out", type=Path, default=Path("models/gbm.txt"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = _data_paths(cfg)
    gcfg = cfg.raw["gbm"]
    fusion_cfg = {
        "agreement_threshold": cfg.raw["label_fusion"]["agreement_threshold"],
        "single_threshold": cfg.raw["label_fusion"]["single_threshold"],
    }

    tile_ids = _resolve_tiles(None, cfg, split="train")
    X_list, y_list, w_list = [], [], []
    feature_names: list[str] | None = None

    for tid in tile_ids:
        print(f"[train] {tid}")
        s2_dir = paths.s2_dir(tid, split="train")
        s2_entries = list_s2_months(s2_dir)
        if not s2_entries:
            print(f"  skip — no S2 data")
            continue
        ref_tif = paths.s2_tif(tid, s2_entries[0][0], s2_entries[0][1], split="train")
        with rasterio.open(ref_tif) as src:
            ref_crs, ref_transform, ref_shape = src.crs, src.transform, (src.height, src.width)

        years = sorted({y for (y, _) in s2_entries})
        year_base, year_last = years[0], years[-1]

        aef_by_year = {}
        for y in (year_base, year_last):
            p = paths.aef_tiff(tid, y, split="train")
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
            print("  skip — no AEF")
            continue
        aef_feats = aef_features(aef_by_year)

        grid = dict(ref_transform=ref_transform, ref_crs=ref_crs, ref_shape=ref_shape)
        s1_base = s1_annual_stats(paths.s1_dir(tid, split="train"), year_base, **grid)
        s1_last = s1_annual_stats(paths.s1_dir(tid, split="train"), year_last, **grid)
        s2_base = s2_annual_stats(s2_dir, year_base, **grid)
        s2_last = s2_annual_stats(s2_dir, year_last, **grid)

        X, names = pack_features(aef_feats, s2_base, s2_last, s1_base, s1_last)
        feature_names = names

        fused = _load_and_fuse_labels(
            tid, paths,
            ref_crs=ref_crs, ref_transform=ref_transform, ref_shape=ref_shape,
            fusion_cfg=fusion_cfg,
        )
        if fused is None:
            continue

        forest = _build_forest_mask_2020(aef_by_year, year_base, s2_base)

        Xs, ys, ws = subsample_pixels(
            X,
            fused.binary, fused.confidence, forest, fused.agree_count,
            max_pos=gcfg["max_pos_per_tile"], max_neg=gcfg["max_neg_per_tile"],
        )
        X_list.append(Xs)
        y_list.append(ys)
        w_list.append(ws)

    if not X_list:
        raise SystemExit("No training data collected.")
    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    w = np.concatenate(w_list, axis=0)
    print(f"[train] dataset: X={X.shape}, positives={int(y.sum())}, negatives={int((y==0).sum())}")

    cfg_gbm = GBMConfig(
        n_estimators=gcfg["n_estimators"],
        learning_rate=gcfg["learning_rate"],
        num_leaves=gcfg["num_leaves"],
        min_child_samples=gcfg["min_child_samples"],
        feature_fraction=gcfg["feature_fraction"],
        bagging_fraction=gcfg["bagging_fraction"],
        bagging_freq=gcfg["bagging_freq"],
    )
    model = PixelGBM(cfg_gbm).fit(X, y, weights=w, feature_names=feature_names)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.out)
    print(f"[train] wrote {args.out}")


if __name__ == "__main__":
    main()

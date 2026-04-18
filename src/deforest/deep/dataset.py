"""Tile → preprocessed patch cache → PyTorch Dataset.

The deep stack never reads raw GeoTIFFs during training — reading Sentinel
time series from disk is the bottleneck on a GPU box. Instead we precompute
per-tile feature + label tensors once, memory-map them on /mnt/scratch, and
extract fixed-size training patches via a lightweight index.

The cache layout under ``cache_dir`` is::

    <tile_id>/
        features.npy    (F, H, W) float16  — AEF + S1/S2 stats, packed
        labels.npy      (H, W)    uint8   — fused binary label
        month.npy       (H, W)    int16   — month-of-change (0 = no event)
        weight.npy      (H, W)    float16 — fused confidence (also loss mask)
        forest.npy      (H, W)    uint8   — 2020 forest mask
        meta.json                         — CRS/transform + feature names

One ``patch_index.jsonl`` at the root lists every valid patch::

    {"tile_id": "...", "y": 0, "x": 256, "positive_fraction": 0.13}

The Dataset draws from ``patch_index`` and performs flip/rot90 augmentation
on the fly. For GPU inference, use :func:`iterate_tile_patches` which walks
one tile end to end without shuffling.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - torch optional
    torch = None  # type: ignore
    Dataset = object  # type: ignore


# ---------------------------------------------------------------------------
# On-disk layout helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CachedTile:
    tile_dir: Path
    features: np.ndarray     # memmap (F, H, W) float16
    labels: np.ndarray       # memmap (H, W)   uint8
    month: np.ndarray        # memmap (H, W)   int16
    weight: np.ndarray       # memmap (H, W)   float16
    forest: np.ndarray       # memmap (H, W)   uint8
    meta: dict

    @classmethod
    def open(cls, tile_dir: Path) -> "CachedTile":
        features = np.load(tile_dir / "features.npy", mmap_mode="r")
        labels = np.load(tile_dir / "labels.npy", mmap_mode="r")
        month = np.load(tile_dir / "month.npy", mmap_mode="r")
        weight = np.load(tile_dir / "weight.npy", mmap_mode="r")
        forest = np.load(tile_dir / "forest.npy", mmap_mode="r")
        meta = json.loads((tile_dir / "meta.json").read_text())
        return cls(tile_dir, features, labels, month, weight, forest, meta)


@dataclass(frozen=True)
class PatchRef:
    tile_id: str
    y: int
    x: int
    positive_fraction: float = 0.0


def _index_path(cache_dir: Path, split: str | None) -> Path:
    """Return the path to the patch index for ``split`` (or the combined one)."""
    if split is None:
        return cache_dir / "patch_index.jsonl"
    return cache_dir / f"patch_index.{split}.jsonl"


def load_patch_index(
    cache_dir: Path, split: str | None = None
) -> list[PatchRef]:
    """Load a patch index.

    ``split=None`` returns the combined index (legacy ``patch_index.jsonl``);
    ``split="train"`` / ``"test"`` returns the per-split file written by
    :func:`write_patch_index`. If the requested per-split file is missing
    we fall back to the combined index so old caches keep working.
    """
    path = _index_path(cache_dir, split)
    if not path.exists() and split is not None:
        path = _index_path(cache_dir, None)
    if not path.exists():
        return []
    out: list[PatchRef] = []
    with path.open("r") as f:
        for line in f:
            d = json.loads(line)
            out.append(
                PatchRef(
                    tile_id=d["tile_id"],
                    y=int(d["y"]),
                    x=int(d["x"]),
                    positive_fraction=float(d.get("positive_fraction", 0.0)),
                )
            )
    return out


def write_patch_index(
    cache_dir: Path, refs: Iterable[PatchRef], *, split: str | None = None
) -> Path:
    """Write a patch index file (combined or per-split)."""
    path = _index_path(cache_dir, split)
    with path.open("w") as f:
        for r in refs:
            f.write(
                json.dumps(
                    {
                        "tile_id": r.tile_id,
                        "y": r.y,
                        "x": r.x,
                        "positive_fraction": r.positive_fraction,
                    }
                )
            )
            f.write("\n")
    return path


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------


class PatchDataset(Dataset):  # type: ignore[misc]
    """Supervised patch dataset.

    Each item is a dict with tensors::

        x           (F, P, P)  float32
        y_change    (P, P)     float32  (0/1)
        y_month     (P, P)     long      (month index, -100 where unlabeled)
        weight      (P, P)     float32   (sample weight per pixel)
        forest      (P, P)     float32   (0/1 — zero-weighted outside)
    """

    def __init__(
        self,
        cache_dir: str | Path,
        patch_size: int = 256,
        *,
        refs: list[PatchRef] | None = None,
        augment: bool = True,
        positive_oversample: float = 2.0,
        rng_seed: int = 1337,
    ):
        if torch is None:
            raise ImportError("PyTorch is required for PatchDataset")
        self.cache_dir = Path(cache_dir)
        self.patch_size = int(patch_size)
        self.augment = bool(augment)
        self._rng = np.random.default_rng(rng_seed)

        self.refs = refs if refs is not None else load_patch_index(self.cache_dir)
        if not self.refs:
            raise FileNotFoundError(f"No patches indexed under {self.cache_dir}")

        if positive_oversample > 1.0:
            self.refs = _oversample_positives(self.refs, positive_oversample)

        self._tiles: dict[str, CachedTile] = {}

    def __len__(self) -> int:
        return len(self.refs)

    def _tile(self, tile_id: str) -> CachedTile:
        t = self._tiles.get(tile_id)
        if t is None:
            t = CachedTile.open(self.cache_dir / tile_id)
            self._tiles[tile_id] = t
        return t

    def __getitem__(self, idx: int) -> dict:
        ref = self.refs[idx]
        tile = self._tile(ref.tile_id)
        p = self.patch_size
        _, h, w = tile.features.shape
        # Clamp the patch origin so we never slice past the tile edge — the
        # collate path requires every sample to be exactly (p, p). Some
        # historical caches contain refs from tiny tiles where ``y + p > h``;
        # without clamping numpy silently truncates and ``torch.stack`` later
        # raises ``stack expects each tensor to be equal size``.
        y = max(0, min(int(ref.y), max(0, h - p)))
        x = max(0, min(int(ref.x), max(0, w - p)))

        feats = np.array(tile.features[:, y : y + p, x : x + p], dtype=np.float32)
        labels = np.array(tile.labels[y : y + p, x : x + p], dtype=np.float32)
        month = np.array(tile.month[y : y + p, x : x + p], dtype=np.int64)
        weight = np.array(tile.weight[y : y + p, x : x + p], dtype=np.float32)
        forest = np.array(tile.forest[y : y + p, x : x + p], dtype=np.float32)

        # Pad if the tile itself is smaller than a patch (degenerate tile).
        if feats.shape[-2:] != (p, p):
            feats = _pad_to(feats, (feats.shape[0], p, p))
            labels = _pad_to(labels, (p, p))
            month = _pad_to(month, (p, p))
            weight = _pad_to(weight, (p, p))
            forest = _pad_to(forest, (p, p))

        # Month head is undefined on no-event pixels — mark as ignore.
        y_month = np.where(labels > 0, month - 1, -100).astype(np.int64)
        # Zero the change loss outside the 2020 forest mask (match inference).
        weight = weight * forest + (1.0 - forest) * 1.0
        # Above keeps a nominal weight so the model still sees non-forest as
        # negative signal; we explicitly re-zero the positives there below.
        labels = labels * forest

        if self.augment:
            feats, labels, y_month, weight, forest = _random_flip_rot(
                feats, labels, y_month, weight, forest, self._rng
            )

        # ``torch.from_numpy`` keeps numpy-owned storage which is NOT resizable;
        # the multi-worker DataLoader with ``pin_memory=True`` then crashes in
        # ``default_collate`` (``Trying to resize storage that is not resizable``).
        # Allocating fresh torch tensors (via ``torch.tensor`` / ``.clone()``)
        # gives torch-owned, resizable storage and fixes the collate path.
        return {
            "x": torch.from_numpy(np.ascontiguousarray(feats)).clone(),
            "y_change": torch.from_numpy(np.ascontiguousarray(labels)).clone(),
            "y_month": torch.from_numpy(np.ascontiguousarray(y_month)).clone(),
            "weight": torch.from_numpy(np.ascontiguousarray(weight)).clone(),
            "forest": torch.from_numpy(np.ascontiguousarray(forest)).clone(),
        }


def _pad_to(arr: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    """Right/bottom-pad ``arr`` with zeros to ``target_shape``."""
    if arr.shape == target_shape:
        return arr
    out = np.zeros(target_shape, dtype=arr.dtype)
    sl = tuple(slice(0, min(a, t)) for a, t in zip(arr.shape, target_shape))
    out[sl] = arr[sl]
    return out


def patch_collate(batch: list[dict]) -> dict:
    """Stack patch dicts without PyTorch's shared-memory fastpath.

    The default ``collate_tensor_fn`` allocates the output via
    ``elem._typed_storage()._new_shared(...).resize_(...)``. On this
    ROCm 2.5.1 build that path raises ``Trying to resize storage that is
    not resizable`` for our tensors because the worker-side shared
    storage is created in a state torch refuses to resize. ``torch.stack``
    on freshly allocated, contiguous tensors avoids the resize call
    entirely while still benefiting from pinned memory transfer.
    """
    keys = batch[0].keys()
    return {k: torch.stack([b[k].contiguous() for b in batch], dim=0) for k in keys}


def _oversample_positives(refs: list[PatchRef], factor: float) -> list[PatchRef]:
    pos = [r for r in refs if r.positive_fraction > 0.002]
    neg = [r for r in refs if r.positive_fraction <= 0.002]
    extra = int(round(len(pos) * (factor - 1)))
    if extra <= 0 or not pos:
        return refs
    rng = np.random.default_rng(0)
    pick = rng.choice(len(pos), size=extra, replace=True)
    return refs + [pos[i] for i in pick]


def _random_flip_rot(
    x: np.ndarray, y: np.ndarray, m: np.ndarray, w: np.ndarray, f: np.ndarray, rng
):
    if rng.random() < 0.5:
        x = np.ascontiguousarray(x[:, :, ::-1])
        y = np.ascontiguousarray(y[:, ::-1])
        m = np.ascontiguousarray(m[:, ::-1])
        w = np.ascontiguousarray(w[:, ::-1])
        f = np.ascontiguousarray(f[:, ::-1])
    if rng.random() < 0.5:
        x = np.ascontiguousarray(x[:, ::-1, :])
        y = np.ascontiguousarray(y[::-1, :])
        m = np.ascontiguousarray(m[::-1, :])
        w = np.ascontiguousarray(w[::-1, :])
        f = np.ascontiguousarray(f[::-1, :])
    k = int(rng.integers(0, 4))
    if k:
        x = np.ascontiguousarray(np.rot90(x, k=k, axes=(1, 2)))
        y = np.ascontiguousarray(np.rot90(y, k=k))
        m = np.ascontiguousarray(np.rot90(m, k=k))
        w = np.ascontiguousarray(np.rot90(w, k=k))
        f = np.ascontiguousarray(np.rot90(f, k=k))
    return x, y, m, w, f


# ---------------------------------------------------------------------------
# Inference patch iterator (tile-wise, deterministic, overlap-blended)
# ---------------------------------------------------------------------------


def iterate_tile_patches(
    tile: CachedTile, patch_size: int, overlap: int
) -> Iterator[tuple[int, int, np.ndarray, np.ndarray]]:
    """Yield overlapping crops covering the whole tile.

    Emits (y, x, features, forest_mask) in row-major order.
    """
    _, h, w = tile.features.shape
    stride = max(1, patch_size - overlap)
    ys = _axis_positions(h, patch_size, stride)
    xs = _axis_positions(w, patch_size, stride)
    for y in ys:
        for x in xs:
            feats = np.asarray(
                tile.features[:, y : y + patch_size, x : x + patch_size], dtype=np.float32
            )
            forest = np.asarray(
                tile.forest[y : y + patch_size, x : x + patch_size], dtype=np.float32
            )
            yield y, x, feats, forest


def _axis_positions(total: int, patch: int, stride: int) -> list[int]:
    if total <= patch:
        return [0]
    positions = list(range(0, total - patch + 1, stride))
    if positions[-1] != total - patch:
        positions.append(total - patch)
    return positions


def hann_window_2d(size: int) -> np.ndarray:
    """A 2D Hann window used for overlap-add stitching during inference."""
    w1 = np.hanning(size).astype(np.float32)
    w = np.outer(w1, w1)
    # Avoid exact zeros at the edges (causes division issues in accumulators).
    w = np.clip(w, 1e-3, None)
    return w

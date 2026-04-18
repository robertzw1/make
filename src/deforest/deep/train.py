"""Training loop for :class:`ChangeUNet` on cached patches.

Designed for a single MI300X (1 GPU, 192 GB HBM):

* **bf16 AMP** via ``torch.autocast`` — MI300X has native bf16 matmul.
* Large batch size (from :func:`runtime.autoscale_defaults`) — we don't need
  DDP because the droplet has a single GPU, but the loop is DDP-ready if we
  ever scale out (``DEFOREST_WORLD_SIZE`` env var would gate it).
* Cosine LR schedule with warmup, AdamW optimizer, gradient clipping.
* **Leave-one-region-out validation** — by default we hold out every tile
  whose MGRS zone matches the last group alphabetically. Override with
  ``--val-tiles``.
* Checkpoints the best model by validation Union-IoU (approximated at the
  patch level) to ``cache_dir/.. /checkpoints``.

The loop does not use distributed data parallel for simplicity — if/when we
scale to multiple MI300X GPUs, wrap ``model`` and ``DataLoader`` in
``DistributedDataParallel`` and ``DistributedSampler`` respectively.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import torch
    from torch import nn
    from torch.optim import AdamW
    from torch.utils.data import DataLoader
except ImportError:  # pragma: no cover - torch optional
    torch = None  # type: ignore

from ..runtime import Hardware, Scales, autoscale_defaults, detect_hardware
from .dataset import PatchDataset, PatchRef, load_patch_index, patch_collate
from .losses import LossWeights, total_loss
from .model import ChangeUNet, ChangeUNetConfig, build_model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def train(
    cache_dir: str | Path,
    checkpoint_dir: str | Path,
    cfg: dict,
    *,
    val_tiles: list[str] | None = None,
    resume: str | Path | None = None,
) -> Path:
    if torch is None:
        raise ImportError("PyTorch is required; install requirements-gpu.txt first")

    cache_dir = Path(cache_dir)
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    hw = detect_hardware()
    scales = autoscale_defaults(hw, patch_size=int(cfg.get("patch_size", 256)))
    scales = _override_from_cfg(scales, cfg)
    _log(f"[deep.train] {hw.summary()}")
    _log(f"[deep.train] scales={scales}")

    # --- data --------------------------------------------------------------
    # Read the train-split index when it exists; otherwise fall back to the
    # combined ``patch_index.jsonl`` (kept for backward compatibility with
    # caches built before the per-split layout existed).
    all_refs = load_patch_index(cache_dir, split="train")
    if not all_refs:
        all_refs = load_patch_index(cache_dir)
    if not all_refs:
        raise FileNotFoundError(
            f"Empty patch cache at {cache_dir}. "
            "Did you run `make preprocess SERVER_CFG=configs/server.yaml`?"
        )
    train_refs, val_refs = _split_refs(all_refs, val_tiles, val_fraction=cfg.get("val_fraction", 0.15))
    _log(f"[deep.train] train={len(train_refs)} patches, val={len(val_refs)} patches")
    if not train_refs:
        raise RuntimeError(
            "Train split is empty after holding out the validation group. "
            "Either run preprocess on more tiles or pass --val-tiles to pin "
            "the held-out set explicitly."
        )

    train_ds = PatchDataset(
        cache_dir,
        patch_size=scales.patch_size,
        refs=train_refs,
        augment=cfg.get("augment", {}).get("flip", True),
        positive_oversample=2.0,
        rng_seed=cfg.get("augment", {}).get("rng_seed", 1337),
    )
    val_ds = PatchDataset(
        cache_dir,
        patch_size=scales.patch_size,
        refs=val_refs,
        augment=False,
        positive_oversample=1.0,
        rng_seed=0,
    )

    # Don't ask DataLoader for a batch larger than the dataset (otherwise
    # ``drop_last=True`` yields zero batches and the loop dies with
    # ``StopIteration``). Clamp once and reflect it back into ``scales`` so
    # later log lines stay honest.
    effective_batch = min(scales.batch_size, max(1, len(train_ds)))
    if effective_batch != scales.batch_size:
        _log(
            f"[deep.train] dataset has only {len(train_ds)} train patches; "
            f"clamping batch_size {scales.batch_size} -> {effective_batch}"
        )
    drop_last = len(train_ds) >= scales.batch_size

    train_loader = DataLoader(
        train_ds,
        batch_size=effective_batch,
        shuffle=True,
        num_workers=scales.num_workers,
        pin_memory=scales.pin_memory,
        persistent_workers=scales.persistent_workers and scales.num_workers > 0,
        prefetch_factor=scales.prefetch_factor if scales.num_workers > 0 else None,
        drop_last=drop_last,
        collate_fn=patch_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=max(1, min(effective_batch, len(val_ds) or 1)),
        shuffle=False,
        num_workers=min(scales.num_workers, 4),
        pin_memory=scales.pin_memory,
        collate_fn=patch_collate,
    )

    # --- model / optim ----------------------------------------------------
    first_batch = next(iter(train_loader))
    in_channels = int(first_batch["x"].shape[1])
    model_cfg = ChangeUNetConfig(
        in_channels=in_channels,
        base_channels=int(cfg.get("base_channels", 64)),
        months=int(cfg.get("months", 72)),
        attn_heads=int(cfg.get("attn_heads", 8)),
        attn_layers=int(cfg.get("attn_layers", 2)),
        dropout=float(cfg.get("dropout", 0.05)),
    )
    model = build_model(model_cfg)
    _log(f"[deep.train] model: in_channels={in_channels} params={_count_params(model)/1e6:.2f}M")

    device = torch.device(scales.torch_device)

    # Tier-2 (deep) training MUST run on the GPU on the MI300X droplet —
    # silently falling back to CPU would burn hours and produce nothing
    # useful. Refuse to start unless either a GPU was detected or the user
    # explicitly opted into CPU training via DEFOREST_DEVICE=cpu.
    require_gpu = os.environ.get("DEFOREST_REQUIRE_GPU", "1").strip() not in {"0", "false", "no"}
    explicit_cpu = os.environ.get("DEFOREST_DEVICE", "").strip().lower() == "cpu"
    if device.type == "cpu" and require_gpu and not explicit_cpu:
        raise RuntimeError(
            "Deep model would train on CPU (device=cpu). The Tier-2 deep "
            "model is GPU-only on this project. Install ROCm/CUDA PyTorch "
            "(`make install-gpu`) or explicitly opt in with "
            "`DEFOREST_DEVICE=cpu DEFOREST_REQUIRE_GPU=0`."
        )

    model = model.to(device)
    # Confirm the placement actually took effect (catches subtle ROCm misconfig
    # where torch.cuda.is_available() is True but moving fails silently).
    p_device = next(model.parameters()).device
    if require_gpu and not explicit_cpu and p_device.type == "cpu":
        raise RuntimeError(
            f"model.to({device}) did not move parameters off CPU "
            f"(saw {p_device}). Refusing to train on CPU."
        )
    _log(
        f"[deep.train] model on device={p_device} "
        f"(torch.cuda.is_available={torch.cuda.is_available()}, "
        f"hip={getattr(torch.version, 'hip', None)})"
    )

    if resume is not None and Path(resume).exists():
        state = torch.load(resume, map_location="cpu")
        model.load_state_dict(state["model"])
        _log(f"[deep.train] resumed from {resume}")

    opt = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 1e-4))

    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * int(cfg["epochs"])
    warmup = int(cfg.get("warmup_steps", 500))

    def lr_at(step: int) -> float:
        if step < warmup:
            return cfg["lr"] * (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return cfg.get("min_lr", 1e-5) + 0.5 * (cfg["lr"] - cfg.get("min_lr", 1e-5)) * (
            1 + math.cos(math.pi * progress)
        )

    amp_dtype = _torch_dtype(scales.amp_dtype)
    use_amp = amp_dtype != torch.float32
    scaler = torch.amp.GradScaler(enabled=use_amp and amp_dtype == torch.float16)

    weights = LossWeights(
        focal_alpha=cfg["loss"]["focal_alpha"],
        focal_gamma=cfg["loss"]["focal_gamma"],
        dice_weight=cfg["loss"]["dice_weight"],
        month_ce_weight=cfg["loss"]["month_ce_weight"],
    )

    best_iou = -1.0
    global_step = 0
    t0 = time.time()
    log_every = max(1, int(cfg.get("log_every_steps", 10)))

    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        ep_logs = {"loss": 0.0, "focal": 0.0, "dice": 0.0, "month_ce": 0.0, "n": 0}
        for batch in train_loader:
            batch = _to_device(batch, device)
            for pg in opt.param_groups:
                pg["lr"] = lr_at(global_step)

            opt.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    out = model(batch["x"])
                    loss, logs = total_loss(out, batch, weights)
                if amp_dtype == torch.float16:
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 1.0))
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 1.0))
                    opt.step()
            else:
                out = model(batch["x"])
                loss, logs = total_loss(out, batch, weights)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.get("grad_clip", 1.0))
                opt.step()

            global_step += 1
            for k, v in logs.items():
                ep_logs[k] += v
            ep_logs["n"] += 1
            if global_step % log_every == 0 or global_step == 1:
                _log(
                    f"[deep.train] step {global_step:>6d} epoch {epoch:02d} "
                    f"loss={logs['loss']:.4f} focal={logs['focal']:.4f} "
                    f"dice={logs['dice']:.4f} ce={logs['month_ce']:.4f} "
                    f"({time.time() - t0:.0f}s)"
                )

        for k in ["loss", "focal", "dice", "month_ce"]:
            ep_logs[k] /= max(1, ep_logs["n"])

        val_iou = _evaluate(model, val_loader, device, amp_dtype)
        elapsed = time.time() - t0
        _log(
            f"[deep.train] epoch {epoch:02d} "
            f"loss={ep_logs['loss']:.4f} focal={ep_logs['focal']:.4f} "
            f"dice={ep_logs['dice']:.4f} ce={ep_logs['month_ce']:.4f} "
            f"val_iou={val_iou:.4f} ({elapsed:.0f}s)"
        )

        ckpt = {
            "model": model.state_dict(),
            "cfg": asdict(model_cfg),
            "epoch": epoch,
            "val_iou": val_iou,
            "in_channels": in_channels,
        }
        torch.save(ckpt, checkpoint_dir / "last.pt")
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(ckpt, checkpoint_dir / "best.pt")
            _log(f"[deep.train] new best: val_iou={val_iou:.4f}")

    return checkpoint_dir / "best.pt"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_device(batch: dict, device) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def _torch_dtype(name: str):
    if torch is None:
        raise ImportError("PyTorch required")
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }[name.lower()]


def _count_params(m: "nn.Module") -> int:
    return sum(p.numel() for p in m.parameters())


def _override_from_cfg(scales: Scales, cfg: dict) -> Scales:
    fields = {}
    for k in ["batch_size", "num_workers", "amp_dtype", "patch_size"]:
        v = cfg.get(k)
        if v is not None:
            fields[k] = v
    if not fields:
        return scales
    return Scales(**{**asdict(scales), **fields})


def _split_refs(
    refs: list[PatchRef],
    val_tiles: list[str] | None,
    val_fraction: float,
) -> tuple[list[PatchRef], list[PatchRef]]:
    if val_tiles:
        val_set = set(val_tiles)
        train = [r for r in refs if r.tile_id not in val_set]
        val = [r for r in refs if r.tile_id in val_set]
        return train, val

    # Leave-one-region-out proxy: group by MGRS prefix (first 5 chars of tile_id).
    groups: dict[str, list[PatchRef]] = {}
    for r in refs:
        key = r.tile_id[:5] if len(r.tile_id) >= 5 else r.tile_id
        groups.setdefault(key, []).append(r)
    if len(groups) == 1:
        # Fall back to a random split.
        rng = np.random.default_rng(0)
        idx = np.arange(len(refs))
        rng.shuffle(idx)
        cut = int(len(refs) * (1 - val_fraction))
        tr = [refs[i] for i in idx[:cut]]
        va = [refs[i] for i in idx[cut:]]
        return tr, va
    val_key = sorted(groups)[-1]
    train = [r for k, rs in groups.items() if k != val_key for r in rs]
    val = groups[val_key]
    return train, val


def _evaluate(model, loader, device, amp_dtype) -> float:
    assert torch is not None
    model.eval()
    inter = 0.0
    union = 0.0
    use_amp = amp_dtype != torch.float32
    with torch.no_grad():
        for batch in loader:
            batch = _to_device(batch, device)
            if use_amp:
                with torch.autocast(device_type=device.type, dtype=amp_dtype):
                    out = model(batch["x"])
            else:
                out = model(batch["x"])
            p = torch.sigmoid(out["change_logits"]) * batch["forest"]
            p_bin = (p > 0.5).float()
            t = batch["y_change"]
            inter += float((p_bin * t).sum())
            union += float(((p_bin + t) > 0).float().sum())
    model.train()
    if union <= 0:
        return 0.0
    return inter / union


def _log(msg: str) -> None:
    print(msg, flush=True)

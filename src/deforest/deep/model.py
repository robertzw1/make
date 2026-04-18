"""ChangeUNet — two-head segmentation network for deforestation detection.

Design
------
* **Encoder** 4 downsampling stages of (GN, SiLU, Conv3x3) × 2 + strided conv
  downsample. 256 → 128 → 64 → 32 → 16 at the default ``patch_size=256``.
* **Bottleneck** ``TransformerBottleneck`` — a lightweight multi-head
  self-attention block over the 16×16 spatial tokens so a pixel's decision
  can depend on neighbourhood context from across the patch. This is the
  part of U-TAE we keep: attention captures global structure efficiently.
* **Decoder** 4 upsampling stages with skip connections (bilinear + 1×1 conv
  instead of transposed conv → fewer checkerboard artefacts).
* **Two heads**
  - ``change_head``  (1-channel sigmoid)   — per-pixel probability.
  - ``month_head``   (M-channel softmax)  — per-pixel month-of-change,
    computed only on pixels predicted positive during inference.

The network is small enough to run comfortably at batch_size=128 on an
MI300X (192 GB HBM) while large enough to beat LightGBM on Union IoU when
properly trained. Channel base = 64 → ~5.8M params with M=72 months.

This module is importable without torch installed (stubs raise on use).
"""

from __future__ import annotations

from dataclasses import dataclass

try:
    import torch
    from torch import nn
    import torch.nn.functional as F

    _TORCH_OK = True
except ImportError:  # pragma: no cover - torch optional
    torch = None  # type: ignore
    F = None  # type: ignore

    class _NnStub:
        """Stand-in for ``torch.nn`` so ``class X(nn.Module)`` doesn't crash
        the module import. Any attempt to actually instantiate a real layer
        raises — callers must install torch (requirements-gpu.txt) first."""

        class Module:  # noqa: D401
            def __init__(self, *a, **k):
                raise ImportError(
                    "PyTorch is not installed — install requirements-gpu.txt"
                )

        def __getattr__(self, name):  # type: ignore[override]
            raise ImportError(
                "PyTorch is not installed — install requirements-gpu.txt "
                f"(accessed nn.{name})"
            )

    nn = _NnStub()  # type: ignore
    _TORCH_OK = False


@dataclass(frozen=True)
class ChangeUNetConfig:
    in_channels: int = 197
    base_channels: int = 64
    depth: int = 4
    months: int = 72
    attn_heads: int = 8
    attn_layers: int = 2
    dropout: float = 0.05


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


def _require_torch() -> None:
    if torch is None:
        raise ImportError(
            "PyTorch is not installed — install via requirements-gpu.txt "
            "before using deforest.deep.model"
        )


def _double_conv(ci: int, co: int, dropout: float) -> "nn.Module":
    _require_torch()
    return nn.Sequential(
        nn.Conv2d(ci, co, 3, padding=1, bias=False),
        nn.GroupNorm(num_groups=min(32, co), num_channels=co),
        nn.SiLU(inplace=True),
        nn.Dropout2d(dropout),
        nn.Conv2d(co, co, 3, padding=1, bias=False),
        nn.GroupNorm(num_groups=min(32, co), num_channels=co),
        nn.SiLU(inplace=True),
    )


class _Down(nn.Module):
    def __init__(self, ci: int, co: int, dropout: float):
        super().__init__()
        self.block = _double_conv(ci, co, dropout)
        self.down = nn.Conv2d(co, co, 3, stride=2, padding=1)

    def forward(self, x):
        skip = self.block(x)
        down = self.down(skip)
        return down, skip


class _Up(nn.Module):
    def __init__(self, ci: int, co: int, dropout: float, *, skip_channels: int | None = None):
        super().__init__()
        # Decoder input is reduced to `co` channels then concatenated with the
        # encoder skip — whose width is generally NOT equal to `co` in this
        # network (e.g. s3 is c*8 while the matching decoder output is c*4).
        # `skip_channels` defaults to `co` to preserve the symmetric-U-Net case.
        sk = co if skip_channels is None else skip_channels
        self.reduce = nn.Conv2d(ci, co, 1)
        self.block = _double_conv(co + sk, co, dropout)

    def forward(self, x, skip):
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.reduce(x)
        x = torch.cat([x, skip], dim=1)
        return self.block(x)


class _TransformerBottleneck(nn.Module):
    """Multi-head self-attention over spatial tokens at bottleneck resolution."""

    def __init__(self, channels: int, heads: int, layers: int, dropout: float):
        super().__init__()
        self.norm = nn.GroupNorm(min(32, channels), channels)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=heads,
            dim_feedforward=channels * 2,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.stack = nn.TransformerEncoder(encoder_layer, num_layers=layers)

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        b, c, h, w = x.shape
        tokens = self.norm(x).reshape(b, c, h * w).permute(0, 2, 1)
        tokens = self.stack(tokens)
        return tokens.permute(0, 2, 1).reshape(b, c, h, w) + x


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------


class ChangeUNet(nn.Module):  # type: ignore[misc]
    def __init__(self, cfg: ChangeUNetConfig):
        super().__init__()
        _require_torch()
        self.cfg = cfg
        c = cfg.base_channels

        self.stem = _double_conv(cfg.in_channels, c, cfg.dropout)
        self.down1 = _Down(c, c * 2, cfg.dropout)
        self.down2 = _Down(c * 2, c * 4, cfg.dropout)
        self.down3 = _Down(c * 4, c * 8, cfg.dropout)
        self.down4 = _Down(c * 8, c * 8, cfg.dropout)

        self.bottleneck = _TransformerBottleneck(
            channels=c * 8,
            heads=cfg.attn_heads,
            layers=cfg.attn_layers,
            dropout=cfg.dropout,
        )

        # Skip widths come from the encoder's `_Down.block` output channels:
        #   s4 = c*8, s3 = c*8, s2 = c*4, s0 = c   (s1 is intentionally unused)
        self.up4 = _Up(c * 8, c * 8, cfg.dropout, skip_channels=c * 8)
        self.up3 = _Up(c * 8, c * 4, cfg.dropout, skip_channels=c * 8)
        self.up2 = _Up(c * 4, c * 2, cfg.dropout, skip_channels=c * 4)
        self.up1 = _Up(c * 2, c, cfg.dropout, skip_channels=c)

        self.change_head = nn.Conv2d(c, 1, kernel_size=1)
        self.month_head = nn.Conv2d(c, cfg.months, kernel_size=1)

    def forward(self, x: "torch.Tensor") -> dict[str, "torch.Tensor"]:
        s0 = self.stem(x)
        d1, s1 = self.down1(s0)
        d2, s2 = self.down2(d1)
        d3, s3 = self.down3(d2)
        d4, s4 = self.down4(d3)

        bott = self.bottleneck(d4)

        u4 = self.up4(bott, s4)
        u3 = self.up3(u4, s3)
        u2 = self.up2(u3, s2)
        u1 = self.up1(u2, s0)

        change_logits = self.change_head(u1).squeeze(1)   # (B, H, W)
        month_logits = self.month_head(u1)                # (B, M, H, W)
        return {"change_logits": change_logits, "month_logits": month_logits}


def build_model(cfg: ChangeUNetConfig) -> ChangeUNet:
    return ChangeUNet(cfg)

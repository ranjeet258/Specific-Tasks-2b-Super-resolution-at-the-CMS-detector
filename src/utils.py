"""
src/utils.py
Visualization, logging, and configuration helpers.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import yaml


# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    """Load YAML config file."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with a clean format."""
    fmt = "%(asctime)s  %(levelname)-8s  %(name)s — %(message)s"
    logging.basicConfig(
        level  = level,
        format = fmt,
        handlers = [logging.StreamHandler(sys.stdout)],
    )


# ──────────────────────────────────────────────────────────────────────────────
# TENSOR → NUMPY HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def channel_composite(img: np.ndarray) -> np.ndarray:
    """
    Collapse (C, H, W) → (H, W) by summing channels.
    Useful for calorimeter visualization where all channels show energy.
    """
    return img.sum(axis=0)


# ──────────────────────────────────────────────────────────────────────────────
# VISUALIZATION
# ──────────────────────────────────────────────────────────────────────────────

CHANNEL_NAMES = ["ECAL", "HCAL", "Track"]
CHANNEL_CMAPS = ["hot", "Blues", "Greens"]


def plot_comparison(
    lr:    np.ndarray,   # (3, 64, 64)
    sr:    np.ndarray,   # (3, 125, 125)
    hr:    np.ndarray,   # (3, 125, 125)
    label: int = 0,
    save_path: Optional[str] = None,
    psnr:  Optional[float] = None,
    ssim:  Optional[float] = None,
) -> plt.Figure:
    """
    3-row × 3-channel comparison: LR | SR | HR per channel,
    with composite sum in a 4th column.
    """
    class_name = "Quark" if label == 0 else "Gluon"
    C = lr.shape[0]

    fig = plt.figure(figsize=(16, 11))
    fig.suptitle(
        f"CMS Super-Resolution — {class_name}"
        + (f"   PSNR={psnr:.2f} dB   SSIM={ssim:.4f}" if psnr else ""),
        fontsize=14, fontweight="bold"
    )

    rows = ["LR (64×64)", "SR (125×125)", "HR (125×125)"]
    images = [lr, sr, hr]
    n_cols = C + 1   # 3 channels + composite

    gs = gridspec.GridSpec(3, n_cols, figure=fig, hspace=0.35, wspace=0.25)

    for row_i, (row_label, img) in enumerate(zip(rows, images)):
        for ch in range(C):
            ax = fig.add_subplot(gs[row_i, ch])
            im = ax.imshow(img[ch], cmap=CHANNEL_CMAPS[ch], aspect="auto")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if row_i == 0:
                ax.set_title(CHANNEL_NAMES[ch], fontsize=10, fontweight="bold")
            ax.set_ylabel(row_label if ch == 0 else "", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])

        # Composite column
        ax_c = fig.add_subplot(gs[row_i, C])
        comp = channel_composite(img)
        im_c = ax_c.imshow(comp, cmap="inferno", aspect="auto")
        plt.colorbar(im_c, ax=ax_c, fraction=0.046, pad=0.04)
        if row_i == 0:
            ax_c.set_title("Σ Channels", fontsize=10, fontweight="bold")
        ax_c.set_xticks([]); ax_c.set_yticks([])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_residual_map(
    sr: np.ndarray,
    hr: np.ndarray,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Pixel-wise residual (SR − HR) per channel."""
    C = hr.shape[0]
    fig, axes = plt.subplots(1, C, figsize=(5 * C, 4))
    fig.suptitle("Residual Map: SR − HR", fontsize=13)

    for c, ax in enumerate(axes):
        res = sr[c] - hr[c]
        vmax = max(abs(res.min()), abs(res.max()))
        im = ax.imshow(res, cmap="bwr", vmin=-vmax, vmax=vmax, aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.046)
        ax.set_title(CHANNEL_NAMES[c])
        ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_energy_profile(
    sr: np.ndarray,
    hr: np.ndarray,
    lr: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Marginal energy projections along η and φ axes.
    Shows whether SR correctly reproduces the jet shower shape.
    """
    C = hr.shape[0]
    fig, axes = plt.subplots(C, 2, figsize=(12, 4 * C))
    fig.suptitle("Energy Profiles (η and φ projections)", fontsize=13)

    for c in range(C):
        # η projection (sum over φ)
        ax_eta = axes[c, 0]
        ax_eta.plot(hr[c].sum(axis=1), "b-",  lw=2,   label="HR (target)")
        ax_eta.plot(sr[c].sum(axis=1), "r--", lw=2,   label="SR (predicted)")
        if lr is not None:
            import cv2
            lr_up = cv2.resize(lr[c], (hr.shape[2], hr.shape[1]),
                               interpolation=cv2.INTER_LINEAR)
            ax_eta.plot(lr_up.sum(axis=1), "g:", lw=1.5, label="LR (bicubic)")
        ax_eta.set_xlabel("η bin"); ax_eta.set_ylabel("Σ Energy")
        ax_eta.set_title(f"{CHANNEL_NAMES[c]} — η projection")
        ax_eta.legend(fontsize=8)

        # φ projection (sum over η)
        ax_phi = axes[c, 1]
        ax_phi.plot(hr[c].sum(axis=0), "b-",  lw=2,   label="HR (target)")
        ax_phi.plot(sr[c].sum(axis=0), "r--", lw=2,   label="SR (predicted)")
        ax_phi.set_xlabel("φ bin"); ax_phi.set_ylabel("Σ Energy")
        ax_phi.set_title(f"{CHANNEL_NAMES[c]} — φ projection")
        ax_phi.legend(fontsize=8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_training_curves(
    log_csv: str,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Read a CSV log and plot training / validation curves."""
    import pandas as pd
    df = pd.read_csv(log_csv)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    fig.suptitle("Training Curves", fontsize=14)

    metrics = [
        ("val/psnr",         "PSNR (dB)",          "blue"),
        ("val/ssim",         "SSIM",                "green"),
        ("val/energy_ratio", "Energy Ratio (≈1.0)", "orange"),
        ("train_G/total",    "G Loss (total)",      "red"),
        ("train_D/loss",     "D Loss",              "purple"),
        ("train_G/energy",   "Energy Loss",         "brown"),
    ]

    for ax, (col, ylabel, color) in zip(axes.flat, metrics):
        if col in df.columns:
            ax.plot(df["epoch"], df[col], color=color, lw=2)
            ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
            ax.set_title(ylabel); ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# DEVICE HELPER
# ──────────────────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger = logging.getLogger(__name__)
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        logging.getLogger(__name__).warning("No GPU found — training on CPU (slow)")
    return device

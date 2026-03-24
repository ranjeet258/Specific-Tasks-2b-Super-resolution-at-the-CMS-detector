#!/usr/bin/env python3
"""
scripts/evaluate.py
===================
Evaluate a trained generator on the test set.
Produces:
  - Full metrics table (PSNR, SSIM, MAE, Energy Ratio, Profile χ², Peak Shift)
  - Qualitative comparison figures (LR | SR | HR)
  - Residual maps
  - Energy profile plots

Usage
-----
  python scripts/evaluate.py \
      --config configs/config.yaml \
      --checkpoint results/checkpoints/best_model.pth \
      --output results/evaluation
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, setup_logging, to_numpy
from src.data_loader import CMSDataModule
from src.models import build_generator
from src.metrics import evaluate_batch, MetricTracker
from src.utils import (
    plot_comparison, plot_residual_map,
    plot_energy_profile,
)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate CMS SR-GAN")
    p.add_argument("--config",      type=str, required=True)
    p.add_argument("--checkpoint",  type=str, required=True)
    p.add_argument("--output",      type=str, default="results/evaluation")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--num-figures", type=int, default=16, help="Visual comparison samples")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    setup_logging()
    logger = logging.getLogger(__name__)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Setup ─────────────────────────────────────────────────────────────
    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dm = CMSDataModule(cfg)
    dm.setup(max_samples=args.max_samples)

    G = build_generator(cfg)
    ckpt = torch.load(args.checkpoint, map_location=device)
    G.load_state_dict(ckpt["generator"])
    G.to(device).eval()
    logger.info(f"Loaded model from epoch {ckpt['epoch']}")

    # ── Quantitative Evaluation ────────────────────────────────────────────
    tracker = MetricTracker()
    logger.info("Running quantitative evaluation on test set …")

    for lr, hr, _ in dm.test_loader:
        lr, hr = lr.to(device), hr.to(device)
        sr = G(lr)
        m  = evaluate_batch(sr, hr)
        tracker.update(m)

    means = tracker.mean()
    logger.info("\n" + "─" * 50)
    logger.info("  TEST SET METRICS")
    logger.info("─" * 50)
    for k, v in means.items():
        logger.info(f"  {k:<20s}: {v:.5f}")
    logger.info("─" * 50)

    # Save metrics to JSON
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(means, f, indent=2)

    # ── Qualitative Figures ─────────────────────────────────────────────
    logger.info("Generating qualitative comparison figures …")
    figs_dir = out_dir / "figures"
    figs_dir.mkdir(exist_ok=True)

    n_fig = args.num_figures
    lr_b, hr_b, y_b = dm.get_sample_batch("test", n=n_fig)
    lr_b, hr_b = lr_b.to(device), hr_b.to(device)
    sr_b = G(lr_b)

    for i in range(min(n_fig, lr_b.size(0))):
        lr_np = to_numpy(lr_b[i])
        sr_np = to_numpy(sr_b[i])
        hr_np = to_numpy(hr_b[i])
        label = int(y_b[i].item())

        m_i = evaluate_batch(sr_b[i:i+1], hr_b[i:i+1])

        plot_comparison(
            lr_np, sr_np, hr_np, label=label,
            psnr=m_i["psnr"], ssim=m_i["ssim"],
            save_path=str(figs_dir / f"comparison_{i:03d}.png"),
        )
        plot_residual_map(
            sr_np, hr_np,
            save_path=str(figs_dir / f"residual_{i:03d}.png"),
        )
        plot_energy_profile(
            sr_np, hr_np, lr_np,
            save_path=str(figs_dir / f"profile_{i:03d}.png"),
        )

    logger.info(f"Figures saved to {figs_dir}")
    logger.info("Evaluation complete.")


if __name__ == "__main__":
    main()

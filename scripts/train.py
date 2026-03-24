#!/usr/bin/env python3
"""
scripts/train.py
================
CLI entry point for training the CMS SR-GAN.

Usage
-----
  python scripts/train.py --config configs/config.yaml
  python scripts/train.py --config configs/config.yaml --max-samples 5000
  python scripts/train.py --config configs/config.yaml --resume results/checkpoints/best_model.pth
"""

import argparse
import logging
import sys
import os

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, setup_logging, get_device
from src.data_loader import CMSDataModule
from src.models import build_generator, build_discriminator, count_parameters
from src.trainer import CMSSRTrainer


def parse_args():
    p = argparse.ArgumentParser(description="Train CMS Super-Resolution GAN")
    p.add_argument("--config",      type=str, required=True,  help="Path to config YAML")
    p.add_argument("--max-samples", type=int, default=None,   help="Limit dataset size (debug)")
    p.add_argument("--resume",      type=str, default=None,   help="Checkpoint to resume from")
    p.add_argument("--gpu",         type=int, default=0,      help="GPU index")
    p.add_argument("--verbose",     action="store_true",       help="Debug logging")
    return p.parse_args()


def main():
    args = parse_args()
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    logger = logging.getLogger(__name__)

    # ── Config ────────────────────────────────────────────────────────────
    cfg = load_config(args.config)
    logger.info(f"Loaded config from {args.config}")

    # ── Device ────────────────────────────────────────────────────────────
    import torch
    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    logger.info(f"Training on: {device}")

    # ── Data ──────────────────────────────────────────────────────────────
    dm = CMSDataModule(cfg)
    dm.setup(max_samples=args.max_samples)

    # ── Models ────────────────────────────────────────────────────────────
    G = build_generator(cfg)
    D = build_discriminator(cfg)
    logger.info(f"Generator     : {count_parameters(G)} parameters")
    logger.info(f"Discriminator : {count_parameters(D)} parameters")

    # ── Trainer ────────────────────────────────────────────────────────────
    trainer = CMSSRTrainer(G, D, cfg, device)

    if args.resume:
        trainer.load_checkpoint(args.resume)
        logger.info(f"Resumed from {args.resume}")

    trainer.fit(dm.train_loader, dm.val_loader)


if __name__ == "__main__":
    main()

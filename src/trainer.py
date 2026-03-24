"""
src/trainer.py  — FIXED VERSION
"""
from __future__ import annotations
import os, time, logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast       # ✅ fixed: was torch.cuda.amp

from models import CMSSRGANGenerator, CMSPatchGAN, build_generator, build_discriminator
from losses import PhysicsGANLoss, LSGANLoss
from metrics import evaluate_batch, MetricTracker

logger = logging.getLogger(__name__)


class CMSSRTrainer:
    def __init__(self, generator, discriminator, cfg, device):
        self.G   = generator.to(device)
        self.D   = discriminator.to(device)
        self.cfg = cfg
        self.dev = device

        tcfg = cfg["training"]

        self.opt_G = torch.optim.Adam(
            self.G.parameters(),
            lr=tcfg["optimizer_G"]["lr"], betas=tcfg["optimizer_G"]["betas"])
        self.opt_D = torch.optim.Adam(
            self.D.parameters(),
            lr=tcfg["optimizer_D"]["lr"], betas=tcfg["optimizer_D"]["betas"])

        total_epochs = tcfg["pretrain_epochs"] + tcfg["gan_epochs"]
        self.sched_G = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt_G, T_max=total_epochs, eta_min=cfg["training"]["scheduler"]["eta_min"])
        self.sched_D = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.opt_D, T_max=total_epochs, eta_min=cfg["training"]["scheduler"]["eta_min"])

        self.g_loss_fn = PhysicsGANLoss(cfg)
        self.d_loss_fn = LSGANLoss()
        self.mse       = nn.MSELoss()

        self.amp    = tcfg.get("amp", True) and device.type == "cuda"
        self.scaler = GradScaler("cuda", enabled=self.amp)  # ✅ fixed: device arg required

        # Logging
        lcfg = cfg.get("logging", {})
        self.log_dir   = Path(lcfg.get("log_dir", "/kaggle/working/logs"))
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_every = lcfg.get("log_every", 50)

        # Optional TensorBoard (graceful fallback)
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=str(self.log_dir))
            self._tb = True
        except Exception:
            self.writer = None
            self._tb = False

        self.ckpt_dir    = Path(tcfg.get("checkpoint_dir", "/kaggle/working/checkpoints"))
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.save_every  = tcfg.get("save_every", 999)
        self.keep_last_n = tcfg.get("keep_last_n", 3)

        self.global_step  = 0
        self.best_psnr    = 0.0
        self._saved_ckpts = []

    def _log_scalar(self, tag, val, step):
        if self._tb and self.writer:
            self.writer.add_scalar(tag, val, step)

    # ── Pre-training ──────────────────────────────────────────────────────
    def pretrain_epoch(self, loader, epoch):
        self.G.train()
        total_loss, n = 0.0, 0   # ✅ fixed: count steps manually (no len() on IterableDataset)

        for step, (lr, hr, _) in enumerate(loader):
            lr, hr = lr.to(self.dev), hr.to(self.dev)
            self.opt_G.zero_grad()
            with autocast("cuda", enabled=self.amp):  # ✅ fixed: device arg
                sr   = self.G(lr)
                loss = self.mse(sr, hr)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.opt_G)
            self.scaler.update()
            total_loss += loss.item()
            n += 1
            self.global_step += 1
            if step % self.log_every == 0:
                self._log_scalar("pretrain/mse", loss.item(), self.global_step)
                logger.info(f"  [PRE e{epoch} step {step:4d}] "
                            f"MSE={loss.item():.5f}  "
                            f"GPU={torch.cuda.memory_allocated()/1e9:.1f}GB")

        return total_loss / max(n, 1)  # ✅ fixed: safe division

    # ── GAN training ──────────────────────────────────────────────────────
    def gan_epoch(self, loader, epoch):
        self.G.train(); self.D.train()
        d_steps = self.cfg["training"].get("d_steps", 1)
        running = {"d_loss":0., "g_total":0., "g_pixel":0.,
                   "g_adv":0., "g_energy":0., "g_profile":0.}
        n = 0

        for step, (lr, hr, _) in enumerate(loader):
            lr, hr = lr.to(self.dev), hr.to(self.dev)

            with torch.no_grad():
                sr_detach = self.G(lr)

            for _ in range(d_steps):
                self.opt_D.zero_grad()
                with autocast("cuda", enabled=self.amp):
                    real_pred = self.D(hr)
                    fake_pred = self.D(sr_detach.detach())
                    d_loss = self.d_loss_fn.discriminator_loss(real_pred, fake_pred)
                self.scaler.scale(d_loss).backward()
                self.scaler.step(self.opt_D)
                self.scaler.update()

            self.opt_G.zero_grad()
            with autocast("cuda", enabled=self.amp):
                sr        = self.G(lr)
                fake_pred = self.D(sr)
                g_loss, breakdown = self.g_loss_fn(sr, hr, fake_pred)
            self.scaler.scale(g_loss).backward()
            self.scaler.step(self.opt_G)
            self.scaler.update()

            running["d_loss"]    += d_loss.item()
            running["g_total"]   += breakdown["total"]
            running["g_pixel"]   += breakdown["pixel"]
            running["g_adv"]     += breakdown["adv"]
            running["g_energy"]  += breakdown["energy"]
            running["g_profile"] += breakdown["profile"]
            n += 1
            self.global_step += 1

            if step % self.log_every == 0:
                for k, v in breakdown.items():
                    self._log_scalar(f"train_G/{k}", v, self.global_step)
                self._log_scalar("train_D/loss", d_loss.item(), self.global_step)
                logger.info(f"  [GAN e{epoch} step {step:4d}] "
                            f"G={breakdown['total']:.4f}  "
                            f"D={d_loss.item():.4f}  "
                            f"GPU={torch.cuda.memory_allocated()/1e9:.1f}GB")

        return {k: v / max(n, 1) for k, v in running.items()}

    # ── Validation ────────────────────────────────────────────────────────
    @torch.no_grad()
    def validate(self, loader, epoch):
        self.G.eval()
        tracker = MetricTracker()
        for lr, hr, _ in loader:
            lr, hr = lr.to(self.dev), hr.to(self.dev)
            with autocast("cuda", enabled=self.amp):
                sr = self.G(lr)
            m = evaluate_batch(sr, hr)
            tracker.update(m)
        means = tracker.mean()
        for k, v in means.items():
            self._log_scalar(f"val/{k}", v, epoch)
        return means

    # ── Main fit ──────────────────────────────────────────────────────────
    def fit(self, train_loader, val_loader):
        tcfg        = self.cfg["training"]
        pretrain_ep = tcfg.get("pretrain_epochs", 3)
        gan_ep      = tcfg.get("gan_epochs", 10)

        logger.info(f"{'='*55}")
        logger.info(f"  Pre-train : {pretrain_ep} epochs  |  GAN : {gan_ep} epochs")
        logger.info(f"  AMP (FP16): {self.amp}  |  Device: {self.dev}")
        logger.info(f"{'='*55}")

        for epoch in range(1, pretrain_ep + 1):
            t0  = time.time()
            mse = self.pretrain_epoch(train_loader, epoch)
            val_m = self.validate(val_loader, epoch)
            self.sched_G.step()
            logger.info(
                f"[PRE {epoch:2d}/{pretrain_ep}] MSE={mse:.5f}  "
                f"PSNR={val_m['psnr']:.2f}  SSIM={val_m['ssim']:.4f}  "
                f"({time.time()-t0:.0f}s)")

        for epoch in range(1, gan_ep + 1):
            t0      = time.time()
            train_m = self.gan_epoch(train_loader, epoch)
            val_m   = self.validate(val_loader, epoch + pretrain_ep)
            self.sched_G.step(); self.sched_D.step()
            logger.info(
                f"[GAN {epoch:2d}/{gan_ep}] "
                f"G={train_m['g_total']:.4f}  D={train_m['d_loss']:.4f}  "
                f"PSNR={val_m['psnr']:.2f}  SSIM={val_m['ssim']:.4f}  "
                f"E={val_m.get('energy_ratio', 0):.3f}  ({time.time()-t0:.0f}s)")
            if val_m["psnr"] > self.best_psnr:
                self.best_psnr = val_m["psnr"]
                self._save_checkpoint(epoch + pretrain_ep, tag="best")
            if epoch % self.save_every == 0:
                self._save_checkpoint(epoch + pretrain_ep)

        if self._tb and self.writer:
            self.writer.close()
        logger.info("✅ Training complete.")

    def _save_checkpoint(self, epoch, tag=None):
        name = f"ckpt_epoch{epoch:03d}.pth" if tag is None else f"{tag}_model.pth"
        path = self.ckpt_dir / name
        torch.save({"epoch": epoch, "generator": self.G.state_dict(),
                    "discriminator": self.D.state_dict(),
                    "opt_G": self.opt_G.state_dict(), "opt_D": self.opt_D.state_dict(),
                    "best_psnr": self.best_psnr, "cfg": self.cfg}, path)
        logger.info(f"  ✓ Checkpoint → {path}")
        if tag is None:
            self._saved_ckpts.append(path)
            if len(self._saved_ckpts) > self.keep_last_n:
                old = self._saved_ckpts.pop(0)
                if old.exists(): old.unlink()

    def load_checkpoint(self, path):
        ckpt = torch.load(path, map_location=self.dev)
        self.G.load_state_dict(ckpt["generator"])
        self.D.load_state_dict(ckpt["discriminator"])
        self.opt_G.load_state_dict(ckpt["opt_G"])
        self.opt_D.load_state_dict(ckpt["opt_D"])
        self.best_psnr = ckpt.get("best_psnr", 0.0)
        logger.info(f"Loaded checkpoint from epoch {ckpt['epoch']}")

"""
src/losses.py
=============
Physics-Informed Loss Functions for CMS Super-Resolution GAN.

Total generator loss:
    L_G = λ_pixel  · MSE(SR, HR)
        + λ_adv    · L_adversarial(D(SR))
        + λ_energy · L_energy_conservation(SR, LR)
        + λ_L1     · ||SR||_1  (sparsity)

Physics motivation
------------------
- Energy Conservation: total energy in SR must match energy in HR.
  Formally: Σ SR_ijk ≈ Σ HR_ijk  per sample per channel.
  We enforce this as a soft constraint using MSE on channel sums.

- Sparsity (L1): calorimeter images are highly sparse — most pixels are
  zero. L1 regularisation encourages the model to preserve this structure
  rather than "hallucinate" energy deposits in empty regions.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedPixelLoss(nn.Module):
    """
    MSE only on non-zero HR pixels + small weight on zero pixels.
    Fixes the 98% sparsity problem — stops model learning all-zeros.
    """
    def __init__(self, nonzero_weight=10.0, zero_weight=0.1):
        super().__init__()
        self.nzw = nonzero_weight
        self.zw  = zero_weight

    def forward(self, sr, hr):
        nonzero_mask = (hr > 0.01).float()   # non-zero HR pixels
        zero_mask    = 1.0 - nonzero_mask

        nz_loss = (nonzero_mask * (sr - hr)**2).sum() / (nonzero_mask.sum() + 1e-6)
        z_loss  = (zero_mask    * (sr - hr)**2).sum() / (zero_mask.sum()    + 1e-6)

        return self.nzw * nz_loss + self.zw * z_loss



# ──────────────────────────────────────────────────────────────────────────────
# ADVERSARIAL LOSSES
# ──────────────────────────────────────────────────────────────────────────────

class LSGANLoss(nn.Module):
    """
    Least-Squares GAN loss (Mao et al., 2017).
    More stable than vanilla BCE and avoids vanishing gradients.

        D loss: E[(D(real) - 1)²] + E[(D(fake))²]
        G loss: E[(D(fake) - 1)²]
    """

    def __init__(self):
        super().__init__()

    def discriminator_loss(
        self,
        real_pred: torch.Tensor,
        fake_pred: torch.Tensor,
    ) -> torch.Tensor:
        real_loss = F.mse_loss(real_pred, torch.ones_like(real_pred))
        fake_loss = F.mse_loss(fake_pred, torch.zeros_like(fake_pred))
        return 0.5 * (real_loss + fake_loss)

    def generator_loss(self, fake_pred: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(fake_pred, torch.ones_like(fake_pred))


# ──────────────────────────────────────────────────────────────────────────────
# PHYSICS LOSSES
# ──────────────────────────────────────────────────────────────────────────────

class EnergyConservationLoss(nn.Module):
    """Physical energy conservation in LINEAR space."""
    def __init__(self, epsilon: float = 1e-6):
        super().__init__()
        self.eps = epsilon

    def forward(self, sr: torch.Tensor, hr: torch.Tensor) -> torch.Tensor:
        sr_linear = torch.expm1(torch.clamp(sr, min=0.0, max=8.0))
        hr_linear = torch.expm1(torch.clamp(hr, min=0.0, max=8.0))
        e_sr = sr_linear.sum(dim=(-2, -1))
        e_hr = hr_linear.sum(dim=(-2, -1))
        return ((e_sr - e_hr).abs() / (e_hr.abs() + self.eps)).mean()

class SparsityLoss(nn.Module):
    """
    L1 sparsity loss.
    Encourages the model to keep near-zero pixels at zero,
    preserving the sparse structure of calorimeter data.
    """

    def forward(self, sr: torch.Tensor) -> torch.Tensor:
        return sr.abs().mean()


class JetProfileLoss(nn.Module):
    """
    Radial jet profile consistency loss.

    Computes the radially-averaged energy profile (projection onto the η axis)
    of SR and HR and penalises deviations. This ensures the generator
    preserves the radial spread of the jet shower — key for quark/gluon
    discrimination.
    """

    def forward(
        self,
        sr: torch.Tensor,   # (B, C, 125, 125)
        hr: torch.Tensor,   # (B, C, 125, 125)
    ) -> torch.Tensor:
        # Marginal projection along η (dim=-2) and φ (dim=-1)
        profile_sr_eta = sr.sum(dim=-1)   # (B, C, H)
        profile_hr_eta = hr.sum(dim=-1)

        profile_sr_phi = sr.sum(dim=-2)   # (B, C, W)
        profile_hr_phi = hr.sum(dim=-2)

        # Normalise by total energy to make it a shape-only constraint
        eps = 1e-6
        norm_eta = profile_hr_eta.sum(dim=-1, keepdim=True) + eps
        norm_phi = profile_hr_phi.sum(dim=-1, keepdim=True) + eps

        loss = (
            F.mse_loss(profile_sr_eta / norm_eta, profile_hr_eta / norm_eta)
            + F.mse_loss(profile_sr_phi / norm_phi, profile_hr_phi / norm_phi)
        )
        return loss * 0.5


# ──────────────────────────────────────────────────────────────────────────────
# COMBINED GENERATOR LOSS
# ──────────────────────────────────────────────────────────────────────────────

class PhysicsGANLoss(nn.Module):
    """
    Combined generator loss:
        L_G = λ_pixel  · MSE
            + λ_adv    · L_adv (LSGAN)
            + λ_energy · L_energy_conservation
            + λ_L1     · L_sparsity
            + λ_profile· L_jet_profile (optional)

    All λ values come from the config YAML.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        lcfg = cfg["loss"]
        self.lam_pixel   = lcfg.get("lambda_pixel",   1.0)
        self.lam_adv     = lcfg.get("lambda_adv",     0.001)
        self.lam_energy  = lcfg.get("lambda_energy",  0.1)
        self.lam_L1      = lcfg.get("lambda_L1",      0.01)
        self.lam_profile = lcfg.get("lambda_profile",  0.05)

        self.adv_loss     = LSGANLoss()
        self.energy_loss  = EnergyConservationLoss()
        self.sparse_loss  = SparsityLoss()
        self.profile_loss = JetProfileLoss()
        self.masked_pixel = MaskedPixelLoss(nonzero_weight=2.0,  zero_weight=0.1)

    def forward(
        self,
        sr: torch.Tensor,
        hr: torch.Tensor,
        fake_pred: torch.Tensor,
    ) -> tuple[torch.Tensor, dict]:
        """
        Returns
        -------
        total_loss : scalar tensor
        breakdown  : dict of individual loss values (for logging)
        """
        l_pixel   = F.mse_loss(sr, hr)
        l_adv     = self.adv_loss.generator_loss(fake_pred)
        l_energy  = self.energy_loss(sr, hr)
        l_sparse  = self.sparse_loss(sr)
        l_profile = self.profile_loss(sr, hr)

        total = (
            self.lam_pixel   * l_pixel
            + self.lam_adv   * l_adv
            + self.lam_energy * l_energy
            + self.lam_L1    * l_sparse
            + self.lam_profile * l_profile
        )

        breakdown = {
            "pixel":   l_pixel.item(),
            "adv":     l_adv.item(),
            "energy":  l_energy.item(),
            "sparse":  l_sparse.item(),
            "profile": l_profile.item(),
            "total":   total.item(),
        }
        return total, breakdown

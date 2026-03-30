"""
src/metrics.py
Evaluation metrics for CMS super-resolution.

Standard image metrics:
  - PSNR  (Peak Signal-to-Noise Ratio)
  - SSIM  (Structural Similarity Index)
  - MAE   (Mean Absolute Error)

Physics-aware metrics:
  - Energy ratio    : Σ SR / Σ HR  (should be ≈ 1.0)
  - Jet mass (m₀)   : reconstructed from pT, η, φ of energy deposits
  - Profile χ²      : chi-squared of radial energy profiles
  - Peak position   : displacement of max-energy pixel (η, φ)
  - Girth           : pT-weighted jet radius (quark/gluon discriminant)
  - Multiplicity    : number of non-zero pixels per channel

Quark/Gluon Discrimination:
  - QuarkGluonClassifier: simple CNN trained on HR, evaluated on SR
  - If SR preserves jet substructure, classifier accuracy on SR ≈ accuracy on HR
"""

from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.metrics import structural_similarity as ssim_fn
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from typing import Dict, Tuple


# ──────────────────────────────────────────────────────────────────────────────
# STANDARD IMAGE METRICS
# ──────────────────────────────────────────────────────────────────────────────

def compute_psnr(sr: np.ndarray, hr: np.ndarray) -> float:
    """PSNR in dB. Arrays: (C,H,W) float32."""
    sr_n, hr_n = _norm01(sr), _norm01(hr)
    return psnr_fn(hr_n, sr_n, data_range=1.0)


def compute_ssim(sr: np.ndarray, hr: np.ndarray) -> float:
    """Mean SSIM across channels. Arrays: (C,H,W) float32."""
    sr_n, hr_n = _norm01(sr), _norm01(hr)
    return float(np.mean([
        ssim_fn(hr_n[c], sr_n[c], data_range=1.0)
        for c in range(sr_n.shape[0])
    ]))


def compute_mae(sr: np.ndarray, hr: np.ndarray) -> float:
    return float(np.abs(sr - hr).mean())


def _norm01(x: np.ndarray) -> np.ndarray:
    mn, mx = x.min(), x.max()
    return np.zeros_like(x) if (mx - mn) < 1e-8 else (x - mn) / (mx - mn)


# ──────────────────────────────────────────────────────────────────────────────
# PHYSICS METRICS
# ──────────────────────────────────────────────────────────────────────────────

def energy_ratio(sr: np.ndarray, hr: np.ndarray) -> float:
    """Σ SR / Σ HR. Ideal = 1.0."""
    e_hr = hr.sum()
    return float(sr.sum() / e_hr) if abs(e_hr) > 1e-8 else float("nan")


def compute_jet_mass(image: np.ndarray,
                     eta_range: Tuple[float,float] = (-1.3, 1.3),
                     phi_range: Tuple[float,float] = (-1.3, 1.3)) -> float:

    # Use first channel (ECAL) as pT proxy
    pT = np.expm1(np.maximum(image[0], 0.0))   # undo log(x+1)

    H, W = pT.shape
    eta_bins = np.linspace(eta_range[0], eta_range[1], H)
    phi_bins = np.linspace(phi_range[0], phi_range[1], W)
    eta_grid, phi_grid = np.meshgrid(eta_bins, phi_bins, indexing='ij')

    # 4-momentum sum
    px = (pT * np.cos(phi_grid)).sum()
    py = (pT * np.sin(phi_grid)).sum()
    pz = (pT * np.sinh(eta_grid)).sum()
    E  = (pT * np.cosh(eta_grid)).sum()

    m2 = E**2 - px**2 - py**2 - pz**2
    return float(np.sqrt(max(m2, 0.0)))


def compute_girth(image: np.ndarray,
                  eta_range: Tuple[float,float] = (-1.3, 1.3),
                  phi_range: Tuple[float,float] = (-1.3, 1.3)) -> float:

    pT = np.expm1(np.maximum(image[0], 0.0))
    H, W = pT.shape
    total_pt = pT.sum()
    if total_pt < 1e-8:
        return 0.0

    eta_bins = np.linspace(eta_range[0], eta_range[1], H)
    phi_bins = np.linspace(phi_range[0], phi_range[1], W)
    eta_grid, phi_grid = np.meshgrid(eta_bins, phi_bins, indexing='ij')

    # Jet axis = pT-weighted centroid
    eta_c = (pT * eta_grid).sum() / total_pt
    phi_c = (pT * phi_grid).sum() / total_pt

    dR = np.sqrt((eta_grid - eta_c)**2 + (phi_grid - phi_c)**2)
    return float((pT / total_pt * dR).sum())


def compute_multiplicity(image: np.ndarray, threshold: float = 0.01) -> float:
 
    active = (image > threshold).sum(axis=(-2, -1))   # (C,)
    return float(active.mean())


def radial_profile(image: np.ndarray) -> np.ndarray:
    """Radially averaged energy profile of a 2D image."""
    H, W = image.shape
    cy, cx = H // 2, W // 2
    max_r = min(cy, cx)
    profile = np.zeros(max_r)
    counts  = np.zeros(max_r)
    ys, xs = np.ogrid[:H, :W]
    r_map = np.sqrt((ys - cy)**2 + (xs - cx)**2).astype(int)
    for r in range(max_r):
        mask = (r_map == r)
        if mask.any():
            profile[r] = image[mask].sum()
            counts[r]  = mask.sum()
    mask_c = counts > 0
    profile[mask_c] /= counts[mask_c]
    return profile


def profile_chi2(sr: np.ndarray, hr: np.ndarray) -> float:
    """χ² between SR and HR radial energy profiles (averaged over channels)."""
    return float(np.mean([
        ((radial_profile(sr[c]) - radial_profile(hr[c]))**2
         / (radial_profile(hr[c]) + 1e-8)).sum()
        for c in range(sr.shape[0])
    ]))


def peak_shift(sr: np.ndarray, hr: np.ndarray) -> Dict[str, float]:
    """Euclidean distance between peak-energy pixel positions."""
    shifts_eta, shifts_phi = [], []
    for c in range(sr.shape[0]):
        idx_sr = np.unravel_index(np.argmax(sr[c]), sr[c].shape)
        idx_hr = np.unravel_index(np.argmax(hr[c]), hr[c].shape)
        shifts_eta.append(abs(idx_sr[0] - idx_hr[0]))
        shifts_phi.append(abs(idx_sr[1] - idx_hr[1]))
    me, mp = float(np.mean(shifts_eta)), float(np.mean(shifts_phi))
    return {"mean_px": float(np.sqrt(me**2 + mp**2)),
            "eta_shift": me, "phi_shift": mp}


# ──────────────────────────────────────────────────────────────────────────────
# BATCH EVALUATION  (all metrics)
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_batch(
    sr_batch: torch.Tensor,   # (B, C, H, W)
    hr_batch: torch.Tensor,   # (B, C, H, W)
) -> Dict[str, float]:
    """Compute all metrics for a batch. Returns dict of mean values."""
    sr_np = sr_batch.detach().cpu().numpy()
    hr_np = hr_batch.detach().cpu().numpy()

    results: Dict[str, list] = {k: [] for k in [
        "psnr", "ssim", "mae",
        "energy_ratio", "profile_chi2", "peak_shift_px",
        "jet_mass_sr", "jet_mass_hr", "jet_mass_err",
        "girth_sr", "girth_hr", "girth_err",
        "multiplicity_sr", "multiplicity_hr",
    ]}

    for i in range(sr_np.shape[0]):
        sr_i, hr_i = sr_np[i], hr_np[i]
        results["psnr"].append(compute_psnr(sr_i, hr_i))
        results["ssim"].append(compute_ssim(sr_i, hr_i))
        results["mae"].append(compute_mae(sr_i, hr_i))
        results["energy_ratio"].append(energy_ratio(sr_i, hr_i))
        results["profile_chi2"].append(profile_chi2(sr_i, hr_i))
        results["peak_shift_px"].append(peak_shift(sr_i, hr_i)["mean_px"])

        # Jet substructure
        m_sr = compute_jet_mass(sr_i)
        m_hr = compute_jet_mass(hr_i)
        g_sr = compute_girth(sr_i)
        g_hr = compute_girth(hr_i)

        results["jet_mass_sr"].append(m_sr)
        results["jet_mass_hr"].append(m_hr)
        results["jet_mass_err"].append(abs(m_sr - m_hr) / (m_hr + 1e-8))
        results["girth_sr"].append(g_sr)
        results["girth_hr"].append(g_hr)
        results["girth_err"].append(abs(g_sr - g_hr) / (g_hr + 1e-8))
        results["multiplicity_sr"].append(compute_multiplicity(sr_i))
        results["multiplicity_hr"].append(compute_multiplicity(hr_i))

    return {k: float(np.nanmean(v)) for k, v in results.items()}


# ──────────────────────────────────────────────────────────────────────────────
# QUARK / GLUON CLASSIFIER  (Gemini point #3)
# ──────────────────────────────────────────────────────────────────────────────

class QuarkGluonClassifier(nn.Module):


    def __init__(self, in_channels: int = 3, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 125 → 62
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 2: 62 → 31
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3: 31 → 15
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d(4)    # → (B, 128, 4, 4)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(x)))


def train_qg_classifier(
    classifier: QuarkGluonClassifier,
    train_loader,
    val_loader,
    device: torch.device,
    epochs: int = 20,
    use_sr: bool = False,
    generator: nn.Module = None,
) -> Dict[str, list]:

    opt = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    history = {"train_acc": [], "val_acc": []}

    if generator is not None:
        generator.eval()

    for epoch in range(1, epochs + 1):
        # ── Train ─────────────────────────────────────────────────────────
        classifier.train()
        correct, total = 0, 0
        for lr, hr, y in train_loader:
            lr, hr, y = lr.to(device), hr.to(device), y.to(device)

            if use_sr and generator is not None:
                with torch.no_grad():
                    inp = generator(lr)
            else:
                inp = hr

            opt.zero_grad()
            logits = classifier(inp)
            loss   = criterion(logits, y)
            loss.backward()
            opt.step()

            correct += (logits.argmax(1) == y).sum().item()
            total   += y.size(0)

        train_acc = correct / total
        history["train_acc"].append(train_acc)

        # ── Validate ───────────────────────────────────────────────────────
        classifier.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for lr, hr, y in val_loader:
                lr, hr, y = lr.to(device), hr.to(device), y.to(device)
                if use_sr and generator is not None:
                    inp = generator(lr)
                else:
                    inp = hr
                logits = classifier(inp)
                correct += (logits.argmax(1) == y).sum().item()
                total   += y.size(0)

        val_acc = correct / total
        history["val_acc"].append(val_acc)
        sch.step()

        if epoch % 5 == 0:
            print(f"  [QG Classifier epoch {epoch:3d}/{epochs}] "
                  f"train={train_acc:.3f}  val={val_acc:.3f}")

    return history


@torch.no_grad()
def evaluate_qg_discrimination(
    classifier: QuarkGluonClassifier,
    test_loader,
    device: torch.device,
    generator: nn.Module = None,
    use_sr: bool = False,
) -> Dict[str, float]:
   
    classifier.eval()
    if generator is not None:
        generator.eval()

    all_preds, all_labels = [], []

    for lr, hr, y in test_loader:
        lr, hr, y = lr.to(device), hr.to(device), y.to(device)
        inp = generator(lr) if (use_sr and generator is not None) else hr
        preds = classifier(inp).argmax(1)
        all_preds.append(preds.cpu())
        all_labels.append(y.cpu())

    preds  = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()

    acc = float((preds == labels).mean())

    # Per-class
    results = {"accuracy": acc}
    for cls, name in [(0, "quark"), (1, "gluon")]:
        tp = ((preds == cls) & (labels == cls)).sum()
        fp = ((preds == cls) & (labels != cls)).sum()
        fn = ((preds != cls) & (labels == cls)).sum()
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        results[f"{name}_precision"] = float(prec)
        results[f"{name}_recall"]    = float(rec)
        results[f"{name}_f1"]        = float(2 * prec * rec / (prec + rec + 1e-8))

    return results


# ──────────────────────────────────────────────────────────────────────────────
# METRIC TRACKER
# ──────────────────────────────────────────────────────────────────────────────

class MetricTracker:

    def __init__(self):
        self._sums:   Dict[str, float] = {}
        self._counts: Dict[str, int]   = {}

    def update(self, metrics: Dict[str, float]):
        for k, v in metrics.items():
            self._sums[k]   = self._sums.get(k, 0.0) + v
            self._counts[k] = self._counts.get(k, 0)  + 1

    def mean(self) -> Dict[str, float]:
        return {k: self._sums[k] / self._counts[k] for k in self._sums}

    def reset(self):
        self._sums.clear(); self._counts.clear()



# ──────────────────────────────────────────────────────────────────────────────
# IMAGE QUALITY METRICS
# ──────────────────────────────────────────────────────────────────────────────

def compute_psnr(sr: np.ndarray, hr: np.ndarray) -> float:
    """
    PSNR in dB.  Arrays: (C, H, W) or (H, W), float32 in any range.
    We normalise to [0, 1] before computing.
    """
    sr_n = _norm01(sr)
    hr_n = _norm01(hr)
    return psnr_fn(hr_n, sr_n, data_range=1.0)


def compute_ssim(sr: np.ndarray, hr: np.ndarray) -> float:
 
    sr_n = _norm01(sr)
    hr_n = _norm01(hr)
    scores = []
    for c in range(sr_n.shape[0]):
        s = ssim_fn(hr_n[c], sr_n[c], data_range=1.0)
        scores.append(s)
    return float(np.mean(scores))


def compute_mae(sr: np.ndarray, hr: np.ndarray) -> float:
    return float(np.abs(sr - hr).mean())


def _norm01(x: np.ndarray) -> np.ndarray:
    mn, mx = x.min(), x.max()
    if mx - mn < 1e-8:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


# ──────────────────────────────────────────────────────────────────────────────
# PHYSICS METRICS
# ──────────────────────────────────────────────────────────────────────────────

def energy_ratio(sr: np.ndarray, hr: np.ndarray) -> float:
  
    e_sr = sr.sum()
    e_hr = hr.sum()
    if abs(e_hr) < 1e-8:
        return float("nan")
    return float(e_sr / e_hr)


def radial_profile(image: np.ndarray) -> np.ndarray:

    H, W = image.shape
    cy, cx = H // 2, W // 2
    max_r = min(cy, cx)
    profile = np.zeros(max_r)
    counts  = np.zeros(max_r)
    for y in range(H):
        for x in range(W):
            r = int(np.sqrt((y - cy)**2 + (x - cx)**2))
            if r < max_r:
                profile[r] += image[y, x]
                counts[r]  += 1
    mask = counts > 0
    profile[mask] /= counts[mask]
    return profile


def profile_chi2(sr: np.ndarray, hr: np.ndarray) -> float:

    chi2s = []
    for c in range(sr.shape[0]):
        p_sr = radial_profile(sr[c])
        p_hr = radial_profile(hr[c])
        denom = p_hr + 1e-8
        chi2s.append(float(((p_sr - p_hr)**2 / denom).sum()))
    return float(np.mean(chi2s))


def peak_shift(sr: np.ndarray, hr: np.ndarray) -> Dict[str, float]:

    shifts_eta, shifts_phi = [], []
    for c in range(sr.shape[0]):
        idx_sr = np.unravel_index(np.argmax(sr[c]), sr[c].shape)
        idx_hr = np.unravel_index(np.argmax(hr[c]), hr[c].shape)
        shifts_eta.append(abs(idx_sr[0] - idx_hr[0]))
        shifts_phi.append(abs(idx_sr[1] - idx_hr[1]))

    mean_eta = float(np.mean(shifts_eta))
    mean_phi = float(np.mean(shifts_phi))
    return {
        "mean_px":   float(np.sqrt(mean_eta**2 + mean_phi**2)),
        "eta_shift": mean_eta,
        "phi_shift": mean_phi,
    }


# ──────────────────────────────────────────────────────────────────────────────
# BATCH EVALUATION
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_batch(
    sr_batch: torch.Tensor,   # (B, C, H, W)
    hr_batch: torch.Tensor,   # (B, C, H, W)
) -> Dict[str, float]:

    sr_np = sr_batch.detach().cpu().numpy()
    hr_np = hr_batch.detach().cpu().numpy()
    B = sr_np.shape[0]

    results: Dict[str, list] = {
        "psnr": [], "ssim": [], "mae": [],
        "energy_ratio": [], "profile_chi2": [],
        "peak_shift_px": [],
    }

    for i in range(B):
        sr_i, hr_i = sr_np[i], hr_np[i]
        results["psnr"].append(compute_psnr(sr_i, hr_i))
        results["ssim"].append(compute_ssim(sr_i, hr_i))
        results["mae"].append(compute_mae(sr_i, hr_i))
        results["energy_ratio"].append(energy_ratio(sr_i, hr_i))
        results["profile_chi2"].append(profile_chi2(sr_i, hr_i))
        results["peak_shift_px"].append(peak_shift(sr_i, hr_i)["mean_px"])

    return {k: float(np.nanmean(v)) for k, v in results.items()}


class MetricTracker:

    def __init__(self):
        self._sums: Dict[str, float] = {}
        self._counts: Dict[str, int] = {}

    def update(self, metrics: Dict[str, float]):
        for k, v in metrics.items():
            self._sums[k]   = self._sums.get(k, 0.0) + v
            self._counts[k] = self._counts.get(k, 0) + 1

    def mean(self) -> Dict[str, float]:
        return {k: self._sums[k] / self._counts[k] for k in self._sums}

    def reset(self):
        self._sums.clear()
        self._counts.clear()

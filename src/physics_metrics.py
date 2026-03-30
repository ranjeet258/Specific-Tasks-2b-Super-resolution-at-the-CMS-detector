"""
src/physics_metrics.py
Physics-aware evaluation metrics for CMS calorimeter super-resolution.

Implements the jet substructure observables that mentors specifically look for:

  1. Jet Mass (m₀)            — invariant mass from calorimeter deposits
  2. Jet Width (girth)        — pT-weighted mean ΔR from jet axis
  3. N-subjettiness (τ₁, τ₂, τ₂₁) — soft-drop substructure
  4. Multiplicity             — number of active cells
  5. pT-Centroid              — energy-weighted centre of mass in η–φ
  6. Quark-Gluon Discriminability — AUC of a linear classifier on SR vs HR features

Physics background
Quark jets: narrow, collimated, low multiplicity, low τ₂₁
Gluon jets: wide, diffuse, high multiplicity, high τ₂₁

If super-resolution is physically meaningful, the *distribution* of these
observables computed on SR images should match those computed on HR images.
A perfect SR model → overlap integral ≈ 1.0 for all observables.
"""

from __future__ import annotations
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional


# ──────────────────────────────────────────────────────────────────────────────
# COORDINATE HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _pixel_coords(H: int, W: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return η and φ coordinate grids for an H×W image.
    Centred at (H/2, W/2), scaled so the full image spans ±1 unit.
    """
    eta = (np.arange(H) - H/2) / (H/2)   # shape (H,)
    phi = (np.arange(W) - W/2) / (W/2)   # shape (W,)
    return np.meshgrid(phi, eta)           # (H,W), (H,W)


def _energy_map(img: np.ndarray) -> np.ndarray:
    """
    Collapse (C,H,W) to (H,W) energy map by summing channels.
    Clip to non-negative (physical: energy cannot be negative).
    """
    return np.clip(img.sum(axis=0), 0, None)


# ──────────────────────────────────────────────────────────────────────────────
# JET OBSERVABLES
# ──────────────────────────────────────────────────────────────────────────────

def jet_pT(img: np.ndarray) -> float:
    """Total transverse energy (sum of all pixels, all channels). Proxy for pT."""
    return float(_energy_map(img).sum())


def jet_centroid(img: np.ndarray) -> Tuple[float, float]:
    """
    Energy-weighted centroid in pixel (η, φ) coordinates.
    Returns (eta_bar, phi_bar) in [-1, 1].
    """
    E = _energy_map(img)
    total = E.sum()
    if total < 1e-8:
        return 0.0, 0.0
    H, W = E.shape
    phi_g, eta_g = _pixel_coords(H, W)
    eta_bar = float((E * eta_g).sum() / total)
    phi_bar = float((E * phi_g).sum() / total)
    return eta_bar, phi_bar


def jet_mass(img: np.ndarray, pT_scale: float = 100.0) -> float:
    """
    Approximate invariant jet mass from calorimeter energy deposits.

    Physics: m² = (ΣE)² - |Σp|²
    In the massless limit with uniform pT per cell:
        m² ≈ ΣΣ_ij  E_i E_j (1 - cos ΔR_ij)
           ≈ pT_total² × <1 - cos ΔR>

    Simplified estimator (used in fast-sim):
        m ≈ pT_scale × √( Σ_cells E_i × ΔR_i² )
    where ΔR_i is the distance of cell i from the jet centroid.

    pT_scale converts normalised pixel energy to GeV (tunable).
    Returns mass in GeV.
    """
    E = _energy_map(img)
    total = E.sum()
    if total < 1e-8:
        return 0.0

    H, W = E.shape
    phi_g, eta_g = _pixel_coords(H, W)
    eta_c, phi_c = jet_centroid(img)

    deta = eta_g - eta_c
    dphi = phi_g - phi_c
    dR2  = deta**2 + dphi**2

    m_sq = pT_scale**2 * (E * dR2).sum()
    return float(np.sqrt(max(m_sq, 0.0)))


def jet_width(img: np.ndarray) -> float:
    """
    Jet width (girth): pT-weighted mean ΔR from jet centroid.
    Gluon jets → wider (larger width) than quark jets.
    """
    E = _energy_map(img)
    total = E.sum()
    if total < 1e-8:
        return 0.0

    H, W = E.shape
    phi_g, eta_g = _pixel_coords(H, W)
    eta_c, phi_c = jet_centroid(img)

    dR = np.sqrt((eta_g - eta_c)**2 + (phi_g - phi_c)**2)
    return float((E * dR).sum() / total)


def jet_multiplicity(img: np.ndarray, threshold: float = 0.01) -> float:
    """
    Number of cells above `threshold` energy (normalised units).
    Gluon jets have higher multiplicity.
    """
    return float((_energy_map(img) > threshold).sum())


def n_subjettiness_tau1(img: np.ndarray) -> float:
    """
    τ₁ (N-subjettiness with N=1): 1-subjet hypothesis.
    τ₁ = Σ_i pT_i × min ΔR(i, axis) / Σ_i pT_i
    Axis = jet centroid (simplified).
    """
    E = _energy_map(img)
    total = E.sum()
    if total < 1e-8:
        return 0.0
    H, W = E.shape
    phi_g, eta_g = _pixel_coords(H, W)
    eta_c, phi_c = jet_centroid(img)
    dR = np.sqrt((eta_g - eta_c)**2 + (phi_g - phi_c)**2)
    return float((E * dR).sum() / (total + 1e-8))


def n_subjettiness_tau2(img: np.ndarray) -> float:
    """
    τ₂ (N-subjettiness with N=2): 2-subjet hypothesis.
    Finds the two highest-energy cells as subjet axes.
    """
    E = _energy_map(img)
    total = E.sum()
    if total < 1e-8:
        return 0.0

    H, W = E.shape
    phi_g, eta_g = _pixel_coords(H, W)

    # Find top-2 energy cells as proxy subjet axes
    flat_idx = np.argsort(E.ravel())[::-1]
    i1 = np.unravel_index(flat_idx[0], E.shape)
    i2 = np.unravel_index(flat_idx[min(1, len(flat_idx)-1)], E.shape)

    eta1, phi1 = eta_g[i1], phi_g[i1]
    eta2, phi2 = eta_g[i2], phi_g[i2]

    dR1 = np.sqrt((eta_g - eta1)**2 + (phi_g - phi1)**2)
    dR2 = np.sqrt((eta_g - eta2)**2 + (phi_g - phi2)**2)
    min_dR = np.minimum(dR1, dR2)

    return float((E * min_dR).sum() / (total + 1e-8))


def n_subjettiness_ratio(img: np.ndarray) -> float:
    """
    τ₂₁ = τ₂ / τ₁: most powerful quark-gluon discriminant.
    Quarks: τ₂₁ → small (one tight subjet)
    Gluons: τ₂₁ → large (two or more subjets)
    """
    t1 = n_subjettiness_tau1(img)
    if t1 < 1e-8:
        return 0.0
    return n_subjettiness_tau2(img) / t1


# ──────────────────────────────────────────────────────────────────────────────
# DISTRIBUTION OVERLAP (for SR quality assessment)
# ──────────────────────────────────────────────────────────────────────────────

def distribution_overlap(sr_vals: np.ndarray, hr_vals: np.ndarray,
                          n_bins: int = 50) -> float:
    """
    Histogram overlap integral between SR and HR observable distributions.
    Returns value in [0, 1] where 1.0 = perfect match.

    For a good SR model:  overlap(jet_mass_SR, jet_mass_HR) → 1.0
    """
    lo = min(sr_vals.min(), hr_vals.min())
    hi = max(sr_vals.max(), hr_vals.max()) + 1e-8
    bins = np.linspace(lo, hi, n_bins + 1)

    h_sr, _ = np.histogram(sr_vals, bins=bins, density=True)
    h_hr, _ = np.histogram(hr_vals, bins=bins, density=True)

    bin_width = bins[1] - bins[0]
    overlap = np.minimum(h_sr, h_hr).sum() * bin_width
    return float(np.clip(overlap, 0.0, 1.0))


def wasserstein1d(sr_vals: np.ndarray, hr_vals: np.ndarray) -> float:
    """
    1D Wasserstein distance (Earth Mover's Distance) between distributions.
    Lower = better agreement. Alternative to overlap.
    """
    s = np.sort(sr_vals)
    h = np.sort(hr_vals)
    # Interpolate to common length
    n = max(len(s), len(h))
    s_i = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(s)), s)
    h_i = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(h)), h)
    return float(np.mean(np.abs(s_i - h_i)))


# ──────────────────────────────────────────────────────────────────────────────
# QUARK-GLUON DISCRIMINABILITY  (the "ultimate proof" per Gemini feedback)
# ──────────────────────────────────────────────────────────────────────────────

def qg_feature_vector(img: np.ndarray) -> np.ndarray:
    """
    5-dimensional physics feature vector for one jet image.
    Used as input to the quark-gluon classifier.
    """
    return np.array([
        jet_width(img),
        jet_multiplicity(img),
        n_subjettiness_ratio(img),
        jet_mass(img),
        jet_pT(img),
    ], dtype=np.float32)


def qg_discriminability_auc(
    sr_imgs:  np.ndarray,   # (N, C, H, W)
    labels:   np.ndarray,   # (N,)  0=quark, 1=gluon
    hr_imgs:  Optional[np.ndarray] = None,  # if provided, compare SR vs HR AUC
) -> Dict[str, float]:
    """
    Train a linear SVM on jet substructure features and measure quark-gluon
    classification AUC.

    If `hr_imgs` is provided, computes AUC for both SR and HR features,
    then reports the ratio (SR_AUC / HR_AUC) — ideally close to 1.0.

    A ratio ≥ 0.95 means SR preserves quark-gluon discriminating information.

    Returns
    -------
    dict with keys: 'sr_auc', 'hr_auc' (optional), 'auc_ratio' (optional)
    """
    try:
        from sklearn.svm import LinearSVC
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import cross_val_score
        from sklearn.pipeline import Pipeline
    except ImportError:
        return {"error": "scikit-learn not installed — pip install scikit-learn"}

    def _features(imgs):
        return np.stack([qg_feature_vector(imgs[i]) for i in range(len(imgs))])

    clf = Pipeline([
        ('scaler', StandardScaler()),
        ('svm',    LinearSVC(max_iter=2000, C=1.0)),
    ])

    # Handle NaN/Inf
    def _clean(X):
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X

    X_sr = _clean(_features(sr_imgs))
    scores_sr = cross_val_score(clf, X_sr, labels, cv=5, scoring='roc_auc')
    result = {'sr_auc': float(scores_sr.mean())}

    if hr_imgs is not None:
        X_hr = _clean(_features(hr_imgs))
        scores_hr = cross_val_score(clf, X_hr, labels, cv=5, scoring='roc_auc')
        result['hr_auc']    = float(scores_hr.mean())
        result['auc_ratio'] = result['sr_auc'] / (result['hr_auc'] + 1e-8)

    return result


# ──────────────────────────────────────────────────────────────────────────────
# BATCH PHYSICS EVALUATION
# ──────────────────────────────────────────────────────────────────────────────

def compute_jet_observables_batch(
    imgs: np.ndarray,   # (N, C, H, W)
) -> Dict[str, np.ndarray]:
    """
    Compute all jet observables for a batch of images.
    Returns dict of 1D arrays of length N.
    """
    N = len(imgs)
    obs = {k: np.zeros(N, dtype=np.float32) for k in
           ['pT', 'mass', 'width', 'multiplicity', 'tau1', 'tau2', 'tau21']}

    for i, img in enumerate(imgs):
        obs['pT'][i]           = jet_pT(img)
        obs['mass'][i]         = jet_mass(img)
        obs['width'][i]        = jet_width(img)
        obs['multiplicity'][i] = jet_multiplicity(img)
        obs['tau1'][i]         = n_subjettiness_tau1(img)
        obs['tau2'][i]         = n_subjettiness_tau2(img)
        obs['tau21'][i]        = n_subjettiness_ratio(img)

    return obs


def physics_evaluation_report(
    sr_batch:  torch.Tensor,    # (N, C, 125, 125)
    hr_batch:  torch.Tensor,    # (N, C, 125, 125)
    labels:    torch.Tensor,    # (N,)
    verbose:   bool = True,
) -> Dict[str, float]:
    """
    Full physics evaluation comparing SR to HR distributions.

    Reports:
      - Overlap integral for each observable (mass, width, τ₂₁, multiplicity)
      - Wasserstein distance for each observable
      - Quark-gluon AUC ratio (SR vs HR)
      - Energy ratio (Σ SR / Σ HR)

    Returns flat dict suitable for logging.
    """
    sr_np = sr_batch.detach().cpu().numpy()
    hr_np = hr_batch.detach().cpu().numpy()
    y_np  = labels.cpu().numpy()

    obs_sr = compute_jet_observables_batch(sr_np)
    obs_hr = compute_jet_observables_batch(hr_np)

    result = {}
    keys   = ['mass', 'width', 'multiplicity', 'tau21']
    for k in keys:
        result[f'overlap_{k}']    = distribution_overlap(obs_sr[k], obs_hr[k])
        result[f'wasserstein_{k}'] = wasserstein1d(obs_sr[k], obs_hr[k])

    # Energy ratio
    e_sr = sr_np.sum(axis=(1,2,3))
    e_hr = hr_np.sum(axis=(1,2,3))
    result['energy_ratio_mean'] = float(np.mean(e_sr / (e_hr + 1e-8)))
    result['energy_ratio_std']  = float(np.std( e_sr / (e_hr + 1e-8)))

    # Quark-gluon discriminability
    qg = qg_discriminability_auc(sr_np, y_np, hr_imgs=hr_np)
    result.update(qg)

    if verbose:
        print("\n" + "─"*55)
        print("  PHYSICS EVALUATION REPORT")
        print("─"*55)
        for k in keys:
            print(f"  {k:<18}  overlap={result[f'overlap_{k}']:.4f}"
                  f"  W1={result[f'wasserstein_{k}']:.4f}")
        print(f"  energy_ratio    mean={result['energy_ratio_mean']:.4f}"
              f"  std={result['energy_ratio_std']:.4f}")
        if 'sr_auc' in result:
            print(f"  QG AUC  SR={result['sr_auc']:.4f}"
                  f"  HR={result.get('hr_auc',0):.4f}"
                  f"  ratio={result.get('auc_ratio',0):.4f}")
        print("─"*55)

    return result

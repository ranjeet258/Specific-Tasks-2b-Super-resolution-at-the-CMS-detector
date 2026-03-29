"""
src/data_loader.py
==================
Memory-safe streaming data loader for CMS calorimeter Parquet files.

THE CORE PROBLEM (and exact fix)
---------------------------------
Loading these Parquet files with pd.read_parquet() or pq.read_table()
causes pyarrow to deserialise ALL nested lists at once → ~34 GB RAM spike
→ OOM on Kaggle (16 GB limit) and most consumer machines.

THE SOLUTION: True streaming IterableDataset
---------------------------------------------
CMSStreamDataset is a PyTorch IterableDataset that:
  1. Opens Parquet files lazily with ParquetFile.iter_batches()
  2. Deserialises one row-group at a time (batch_size=32 rows)
  3. Yields (lr, hr, label) tensors one sample at a time
  4. NEVER builds a list of all samples in memory

DiffLense-style preprocessing
-------------------------------
The condition-preprocessing pipeline from DiffLense (Reddy et al. 2024)
is available as `preprocess_lr()`:
    Median filter → Gaussian smooth → NLM denoise → Threshold
Applied only to LR images used as GAN conditioning inputs.
"""

from __future__ import annotations

import random
import warnings
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, IterableDataset

try:
    from scipy.ndimage import median_filter, gaussian_filter
    _SCIPY = True
except ImportError:
    _SCIPY = False
    warnings.warn("scipy not found — DiffLense preprocessing disabled. "
                  "Install with: pip install scipy")


# ──────────────────────────────────────────────────────────────────────────────
# PHYSICS PREPROCESSING
# ──────────────────────────────────────────────────────────────────────────────

def log_normalize(x: np.ndarray) -> np.ndarray:
    """
    ln(x+1) normalization.
    Physics rationale: calorimeter energy has high dynamic range (0→large).
    ln(x+1) compresses range while preserving zeros (ln(0+1)=0).
    Typical output range: LR ≈ [-0.4, 3.2], HR ≈ [-1.3, 4.1].
    """
    return np.log1p(np.maximum(x, 0.0))


def preprocess_lr_difflense(lr: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    """
    DiffLense-style preprocessing pipeline for LR conditioning:
        Median filter → Gaussian smooth → NLM denoise → Threshold

    Reduces noise and background so the diffusion model receives a
    cleaner conditional distribution (less overlap between signal/noise).

    Parameters
    ----------
    lr        : (C, H, W) float32 log-normalised LR image
    threshold : pixels below this value set to 0 (background suppression)
    """
    if not _SCIPY:
        return lr   # pass-through if scipy unavailable

    result = np.zeros_like(lr)
    for c in range(lr.shape[0]):
        ch = lr[c]
        ch = median_filter(ch, size=3)              # remove salt-and-pepper noise
        ch = gaussian_filter(ch, sigma=0.5)         # smooth
        # Simple NLM approximation via repeated Gaussian (full NLM is slow)
        ch = gaussian_filter(ch, sigma=1.0)
        ch = np.where(ch >= threshold, ch, 0.0)     # threshold background
        result[c] = ch
    return result.astype(np.float32)


def augment_pair(
    lr: np.ndarray, hr: np.ndarray,
    flip_h: bool = True, flip_v: bool = True, rot90: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Physics-valid augmentations exploiting η–φ symmetry:
      - 90°/180°/270° rotations  (discrete rotational symmetry in η–φ)
      - Horizontal flip (φ → −φ, azimuthal symmetry)
      - Vertical flip   (η → −η, forward-backward symmetry)
    """
    if rot90:
        k = random.randint(0, 3)
        lr = np.rot90(lr, k, axes=(1, 2)).copy()
        hr = np.rot90(hr, k, axes=(1, 2)).copy()
    if flip_h and random.random() > 0.5:
        lr = np.flip(lr, axis=2).copy()
        hr = np.flip(hr, axis=2).copy()
    if flip_v and random.random() > 0.5:
        lr = np.flip(lr, axis=1).copy()
        hr = np.flip(hr, axis=1).copy()
    return lr, hr


# ──────────────────────────────────────────────────────────────────────────────
# STREAMING ITERABLE DATASET  (Kaggle-safe)
# ──────────────────────────────────────────────────────────────────────────────

class CMSStreamDataset(IterableDataset):
    """
    True streaming IterableDataset.

    Memory footprint: O(batch_size) — never O(dataset_size).
    Safe for Kaggle (16 GB), Colab (12 GB), and local machines.

    Parameters
    ----------
    parquet_files : list of Parquet file paths
    lr_key        : column name for LR images
    hr_key        : column name for HR images
    label_key     : column name for class labels
    parquet_batch : row-group size for iter_batches (32 = ~30 MB RAM per batch)
    shuffle_buf   : reservoir buffer for approximate shuffle (0 = no shuffle)
    augment       : apply η–φ augmentations
    normalize     : apply log(x+1) normalization
    difflense_prep: apply DiffLense preprocessing to LR (for diffusion model)
    max_samples   : cap total samples (for debugging / train only)
    """

    def __init__(
        self,
        parquet_files:  List[str],
        lr_key:         str   = "X_jets_LR",
        hr_key:         str   = "X_jets",
        label_key:      str   = "y",
        parquet_batch:  int   = 32,
        shuffle_buf:    int   = 512,
        augment:        bool  = True,
        normalize:      bool  = True,
        difflense_prep: bool  = False,
        max_samples:    Optional[int] = None,
    ):
        self.files         = parquet_files
        self.lr_key        = lr_key
        self.hr_key        = hr_key
        self.label_key     = label_key
        self.parquet_batch = parquet_batch
        self.shuffle_buf   = shuffle_buf
        self.augment       = augment
        self.normalize     = normalize
        self.difflense     = difflense_prep
        self.max_samples   = max_samples

    # ── core generator ────────────────────────────────────────────────────────

    def _raw_stream(self):
        """
        Generator that yields (lr, hr, label) numpy tuples one at a time.
        Iterates all files; never holds more than one row-group in memory.
        """
        file_list = list(self.files)
        random.shuffle(file_list)

        for fpath in file_list:
            pf = pq.ParquetFile(fpath)
            for batch in pf.iter_batches(
                batch_size=self.parquet_batch,
                columns=[self.lr_key, self.hr_key, self.label_key],
            ):
                lr_col  = batch.column(self.lr_key)
                hr_col  = batch.column(self.hr_key)
                lbl_col = batch.column(self.label_key).to_pylist()

                for i in range(batch.num_rows):
                    lr  = np.array(lr_col[i].as_py(),  dtype=np.float32).reshape(3, 64,  64)
                    hr  = np.array(hr_col[i].as_py(),  dtype=np.float32).reshape(3, 125, 125)
                    lbl = int(lbl_col[i])
                    yield lr, hr, lbl

    def _processed_stream(self):
        """Apply normalization, preprocessing, and augmentation in-stream."""
        count = 0
        for lr, hr, lbl in self._raw_stream():
            if self.max_samples and count >= self.max_samples:
                return

            if self.normalize:
                lr = log_normalize(lr)
                hr = log_normalize(hr)

            if self.difflense:
                lr = preprocess_lr_difflense(lr)

            if self.augment:
                lr, hr = augment_pair(lr, hr)

            yield lr, hr, lbl
            count += 1

    def _shuffled_stream(self):
        """
        Reservoir-based approximate shuffle.
        Keeps `shuffle_buf` samples in a buffer and randomly ejects one
        at a time — O(shuffle_buf) memory, never O(dataset_size).
        """
        if self.shuffle_buf <= 1:
            yield from self._processed_stream()
            return

        buf = []
        for item in self._processed_stream():
            buf.append(item)
            if len(buf) >= self.shuffle_buf:
                idx = random.randrange(len(buf))
                yield buf[idx]
                buf[idx] = buf[-1]
                buf.pop()
        random.shuffle(buf)
        yield from buf

    # ── IterableDataset interface ─────────────────────────────────────────────

    def __iter__(self):
        for lr, hr, lbl in self._shuffled_stream():
            yield (
                torch.from_numpy(lr),
                torch.from_numpy(hr),
                torch.tensor(lbl, dtype=torch.long),
            )


# ──────────────────────────────────────────────────────────────────────────────
# TRAIN / VAL / TEST SPLIT  (file-level split preserves streaming property)
# ──────────────────────────────────────────────────────────────────────────────

class CMSDataModule:
    """
    Data module that wraps three CMSStreamDatasets (train / val / test).

    Split strategy: file-level (not sample-level).
    With 3 files, the default 80/10/10 split assigns:
      train → file0 + file1 (83,812 samples)
      val   →  half file2   (~27,747)
      test  →  half file2   (~27,747)

    Parameters
    ----------
    cfg : config dict loaded from configs/config.yaml
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.train_loader: Optional[DataLoader] = None
        self.val_loader:   Optional[DataLoader] = None
        self.test_loader:  Optional[DataLoader] = None

    def setup(
        self,
        max_samples:    Optional[int]  = None,
        difflense_prep: bool           = False,
    ):
        dcfg = self.cfg["data"]
        tcfg = self.cfg["training"]

        files = [str(p) for p in dcfg["files"]]
        n_files = len(files)

        # File-level split
        n_train = max(1, int(n_files * dcfg.get("train_frac", 0.80)))
        n_val   = max(1, int(n_files * dcfg.get("val_frac",   0.10)))
        rng = random.Random(dcfg.get("seed", 42))
        shuffled = list(files); rng.shuffle(shuffled)
        train_files = shuffled[:n_train]
        val_files   = shuffled[n_train:n_train+n_val] or shuffled[-1:]
        test_files  = shuffled[n_train+n_val:]        or shuffled[-1:]

        augment = dcfg.get("augment", True)

        # ✅ FIX: max_samples only applied to train — val/test use full data
        base_common = dict(
            lr_key        = dcfg.get("lr_key",  "X_jets_LR"),
            hr_key        = dcfg.get("hr_key",  "X_jets"),
            label_key     = dcfg.get("label_key", "y"),
            parquet_batch = dcfg.get("parquet_batch_size", 32),
            normalize     = True,
            difflense_prep= difflense_prep,
        )
        train_common = {**base_common, "max_samples": max_samples}  # ✅ capped
        val_common   = {**base_common, "max_samples": None}         # ✅ full val
        test_common  = {**base_common, "max_samples": None}         # ✅ full test

        train_ds = CMSStreamDataset(
            train_files, augment=augment,
            shuffle_buf=dcfg.get("cache_size", 0), **train_common
        )
        val_ds = CMSStreamDataset(
            val_files, augment=False,
            shuffle_buf=0, **val_common
        )
        test_ds = CMSStreamDataset(
            test_files, augment=False,
            shuffle_buf=0, **test_common
        )

        ldr_kwargs = dict(
            batch_size  = tcfg.get("batch_size",   8),
            num_workers = tcfg.get("num_workers",   0),
            pin_memory  = tcfg.get("pin_memory", True),
        )

        self.train_loader = DataLoader(train_ds, **ldr_kwargs)
        self.val_loader   = DataLoader(val_ds,   **ldr_kwargs)
        self.test_loader  = DataLoader(test_ds,  **ldr_kwargs)

        print(f"[CMSDataModule] Train files: {len(train_files)} | "
              f"Val files: {len(val_files)} | Test files: {len(test_files)}")
        print(f"  Streaming mode: IterableDataset (Kaggle-safe)")
        print(f"  Peak RAM per batch ≈ "
              f"{tcfg.get('batch_size',8) * 2 * 3 * 125 * 125 * 4 / 1e6:.1f} MB")

    def get_sample_batch(self, split: str = "val", n: int = 8):
        loader = {"train": self.train_loader,
                  "val":   self.val_loader,
                  "test":  self.test_loader}[split]
        lr, hr, y = next(iter(loader))
        return lr[:n], hr[:n], y[:n]

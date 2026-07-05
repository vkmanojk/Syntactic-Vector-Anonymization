"""
Mutual information estimation using MINDE.

This module provides a lightweight wrapper around the MINDE estimator
used in the thesis.

It assumes that the original MINDE repository has been cloned into the
project root as:

project/
├── src/
├── wrappers/
├── notebooks/
└── minde/
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader


class MINDataset(Dataset):
    """
    Dataset wrapper for MINDE.
    """

    def __init__(
        self,
        original_embeddings: np.ndarray,
        anonymized_embeddings: np.ndarray,
    ) -> None:

        self.original = torch.tensor(
            original_embeddings,
            dtype=torch.float32,
        )

        self.anonymized = torch.tensor(
            anonymized_embeddings,
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        return len(self.original)

    def __getitem__(self, index: int):

        return {
            "x": self.original[index],
            "y": self.anonymized[index],
        }


def estimate_mutual_information(
    original_embeddings: np.ndarray,
    anonymized_embeddings: np.ndarray,
    *,
    normalize: bool = True,
    batch_size: int = 512,
    max_epochs: int = 150,
    learning_rate: float = 1e-3,
    random_seed: int = 42,
):
    """
    Estimate mutual information using MINDE.

    Parameters
    ----------
    original_embeddings : np.ndarray
        Original embedding matrix.

    anonymized_embeddings : np.ndarray
        Anonymized embedding matrix.

    normalize : bool, default=True
        Whether to standardize the embeddings before estimation.

    batch_size : int, default=512
        Training batch size.

    max_epochs : int, default=150
        Number of training epochs.

    learning_rate : float, default=1e-3
        Learning rate.

    random_seed : int, default=42
        Random seed.

    Returns
    -------
    tuple[float, float]
        Estimated (mi, mi_sigma).
    """

    if original_embeddings.shape != anonymized_embeddings.shape:
        raise ValueError(
            "Original and anonymized embeddings must have the same shape."
        )

    # ---------------------------------------------------------
    # Locate the external MINDE repository
    # ---------------------------------------------------------

    project_root = Path(__file__).resolve().parent.parent
    minde_root = project_root / "minde"

    if not minde_root.exists():
        raise FileNotFoundError(
            "MINDE repository not found.\n\n"
            "Clone the original MINDE repository into:\n\n"
            f"{minde_root}\n"
        )

    # Temporarily expose MINDE's src package
    sys.path.insert(0, str(minde_root))

    try:

        from src.libs.minde import MINDE
        from src.scripts.helper import get_default_config

        original_embeddings = np.asarray(
            original_embeddings,
            dtype=np.float32,
        )

        anonymized_embeddings = np.asarray(
            anonymized_embeddings,
            dtype=np.float32,
        )

        if normalize:

            scaler = StandardScaler()

            original_embeddings = scaler.fit_transform(
                original_embeddings
            )

            anonymized_embeddings = scaler.transform(
                anonymized_embeddings
            )

        dataset = MINDataset(
            original_embeddings,
            anonymized_embeddings,
        )

        args = get_default_config()

        # MINDE configuration
        args.type = "c"
        args.importance_sampling = True
        args.arch = "mlp"
        args.use_ema = True

        args.max_epochs = max_epochs
        args.warmup_epochs = 5
        args.test_epoch = 5

        args.lr = learning_rate
        args.bs = batch_size

        args.accelerator = (
            "gpu"
            if torch.cuda.is_available()
            else "cpu"
        )

        args.devices = 1

        args.out_dir = "./outputs"
        args.benchmark = "embedding_mi"
        args.seed = random_seed
        args.mc_iter = 20

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
        )

        model = MINDE(
            args,
            var_list={
                "x": original_embeddings.shape[1],
                "y": anonymized_embeddings.shape[1],
            },
            gt=None,
        )

        model.fit(loader, loader)

        mi, mi_sigma = model.compute_mi()

        return float(mi), float(mi_sigma)

    finally:
        # Remove MINDE path to avoid conflicts with this project's src package
        if str(minde_root) in sys.path:
            sys.path.remove(str(minde_root))


__all__ = [
    "estimate_mutual_information",
]
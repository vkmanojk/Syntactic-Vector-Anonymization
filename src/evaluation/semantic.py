"""
Semantic consistency evaluation.

This module provides functions for measuring semantic consistency
between original and anonymized embeddings using cosine similarity.
"""

from __future__ import annotations

import numpy as np


def semantic_consistency(
    original_embeddings: np.ndarray,
    anonymized_embeddings: np.ndarray,
    normalize: bool = True,
) -> float:
    """
    Compute Semantic Consistency as the average cosine similarity between
    corresponding original and anonymized embeddings.

    Parameters
    ----------
    original_embeddings : np.ndarray
        Original embedding matrix of shape
        (n_samples, embedding_dimension).

    anonymized_embeddings : np.ndarray
        Anonymized embedding matrix having the same shape.

    normalize : bool, default=True
        Whether to L2-normalize embeddings before computing cosine
        similarity.

    Returns
    -------
    float
        Average cosine similarity.

    Raises
    ------
    ValueError
        If the two matrices have different shapes.
    """

    original_embeddings = np.asarray(original_embeddings, dtype=float)
    anonymized_embeddings = np.asarray(anonymized_embeddings, dtype=float)

    if original_embeddings.shape != anonymized_embeddings.shape:
        raise ValueError(
            "Original and anonymized embeddings must have the same shape."
        )

    if normalize:
        original_embeddings = original_embeddings / (
            np.linalg.norm(original_embeddings, axis=1, keepdims=True) + 1e-8
        )

        anonymized_embeddings = anonymized_embeddings / (
            np.linalg.norm(anonymized_embeddings, axis=1, keepdims=True) + 1e-8
        )

    return float(
        np.mean(
            np.sum(
                original_embeddings * anonymized_embeddings,
                axis=1,
            )
        )
    )


__all__ = ["semantic_consistency"]
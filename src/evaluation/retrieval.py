"""
Retrieval consistency evaluation.

This module provides functions for evaluating retrieval consistency
between original and anonymized embeddings using Recall@k.
"""

from __future__ import annotations

import numpy as np


def retrieval_consistency(
    original_embeddings: np.ndarray,
    anonymized_embeddings: np.ndarray,
    k: int = 10,
    batch_size: int = 512,
    normalize: bool = True,
) -> float:
    """
    Compute Retrieval Consistency (Recall@k).

    Retrieval consistency is measured as the average overlap between the
    k-nearest neighbours of each embedding in the original and
    anonymized embedding spaces.

    Parameters
    ----------
    original_embeddings : np.ndarray
        Original embedding matrix of shape
        (n_samples, embedding_dimension).

    anonymized_embeddings : np.ndarray
        Anonymized embedding matrix of identical shape.

    k : int, default=10
        Number of nearest neighbours.

    batch_size : int, default=512
        Batch size used during similarity computation to reduce memory
        usage.

    normalize : bool, default=True
        Whether to L2-normalize embeddings before computing cosine
        similarity.

    Returns
    -------
    float
        Mean Recall@k.

    Raises
    ------
    ValueError
        If the input matrices have different shapes.
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

    n_samples = original_embeddings.shape[0]

    if k >= n_samples:
        raise ValueError(
            "k must be smaller than the number of embeddings."
        )

    recalls = []

    for start in range(0, n_samples, batch_size):

        end = min(start + batch_size, n_samples)

        similarity_original = (
            original_embeddings[start:end]
            @ original_embeddings.T
        )

        similarity_anonymized = (
            anonymized_embeddings[start:end]
            @ anonymized_embeddings.T
        )

        for i in range(end - start):
            similarity_original[i, start + i] = -1
            similarity_anonymized[i, start + i] = -1

        original_neighbors = np.argsort(
            -similarity_original,
            axis=1,
        )[:, :k]

        anonymized_neighbors = np.argsort(
            -similarity_anonymized,
            axis=1,
        )[:, :k]

        for i in range(end - start):

            recall = (
                len(
                    set(original_neighbors[i])
                    & set(anonymized_neighbors[i])
                )
                / k
            )

            recalls.append(recall)

    return float(np.mean(recalls))


__all__ = ["retrieval_consistency"]
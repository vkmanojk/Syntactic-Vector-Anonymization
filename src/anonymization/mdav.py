"""
MDAV-C (Maximum Distance to Average Vector with Centroid Replacement)

This module implements the deterministic MDAV clustering algorithm used
for syntactic anonymization of vector embeddings.

Given an embedding matrix and an anonymity parameter k, the algorithm
partitions the embeddings into groups of approximately k records and
replaces every embedding within a group with the group's centroid.
"""

from __future__ import annotations

import numpy as np


def mdav_c(
    embeddings: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    Perform MDAV-C anonymization using centroid replacement.

    Parameters
    ----------
    embeddings : np.ndarray
        Input embedding matrix of shape (n_samples, embedding_dimension).

    k : int
        Desired anonymity parameter. Each cluster will contain
        approximately k embeddings.

    Returns
    -------
    np.ndarray
        An anonymized embedding matrix of the same shape as the input.

    Raises
    ------
    ValueError
        If k < 2 or the number of embeddings is smaller than k.
    """

    if k < 2:
        raise ValueError("k must be at least 2.")

    embeddings = np.asarray(embeddings, dtype=float)

    n_samples = embeddings.shape[0]

    if n_samples < k:
        raise ValueError(
            "Number of embeddings must be greater than or equal to k."
        )

    assigned = np.zeros(n_samples, dtype=bool)
    anonymized = np.zeros_like(embeddings)

    all_indices = np.arange(n_samples)

    while np.sum(~assigned) >= 2 * k:

        # Remaining (unassigned) embeddings
        remaining_indices = all_indices[~assigned]
        remaining_embeddings = embeddings[remaining_indices]

        # Compute centroid of remaining embeddings
        centroid = remaining_embeddings.mean(axis=0)

        # Select the embedding farthest from the centroid
        distances = np.linalg.norm(
            remaining_embeddings - centroid,
            axis=1,
        )
        first_seed = remaining_indices[np.argmax(distances)]

        # Select the embedding farthest from the first seed
        distances = np.linalg.norm(
            remaining_embeddings - embeddings[first_seed],
            axis=1,
        )
        second_seed = remaining_indices[np.argmax(distances)]

        # Form first cluster around the first seed
        distances = np.linalg.norm(
            remaining_embeddings - embeddings[first_seed],
            axis=1,
        )

        first_cluster = remaining_indices[np.argsort(distances)[:k]]

        anonymized[first_cluster] = embeddings[first_cluster].mean(axis=0)
        assigned[first_cluster] = True

        # Stop if fewer than k records remain
        if np.sum(~assigned) < k:
            break

        # Update remaining records
        remaining_indices = all_indices[~assigned]
        remaining_embeddings = embeddings[remaining_indices]

        # Form second cluster around the second seed
        distances = np.linalg.norm(
            remaining_embeddings - embeddings[second_seed],
            axis=1,
        )

        second_cluster = remaining_indices[np.argsort(distances)[:k]]

        anonymized[second_cluster] = embeddings[second_cluster].mean(axis=0)
        assigned[second_cluster] = True

    # Replace any remaining embeddings with their centroid
    remaining_indices = all_indices[~assigned]

    if len(remaining_indices) > 0:
        anonymized[remaining_indices] = (
            embeddings[remaining_indices].mean(axis=0)
        )

    return anonymized


__all__ = ["mdav_c"]
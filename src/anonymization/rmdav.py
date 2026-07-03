"""
RMDAV-M (Randomized Maximum Distance to Average Vector with Medoid Replacement)

This module implements the randomized MDAV clustering algorithm used for
syntactic anonymization of vector embeddings.

Compared to MDAV-C, RMDAV-M introduces randomness during cluster formation
by selecting representatives from the top-r farthest candidates and
replaces each cluster with its medoid (an actual embedding) rather than
its centroid.
"""

from __future__ import annotations

import numpy as np


def _compute_medoid(cluster: np.ndarray) -> np.ndarray:
    """
    Compute the medoid of a cluster.

    Parameters
    ----------
    cluster : np.ndarray
        Cluster embeddings of shape (n_samples, embedding_dimension).

    Returns
    -------
    np.ndarray
        Medoid embedding.
    """
    distance_matrix = np.linalg.norm(
        cluster[:, None, :] - cluster[None, :, :],
        axis=2,
    )

    medoid_index = np.argmin(distance_matrix.sum(axis=1))
    return cluster[medoid_index]


def rmdav_m(
    embeddings: np.ndarray,
    k: int,
    top_r: int = 5,
    random_state: int = 42,
) -> np.ndarray:
    """
    Perform RMDAV-M anonymization.

    Parameters
    ----------
    embeddings : np.ndarray
        Input embedding matrix of shape
        (n_samples, embedding_dimension).

    k : int
        Desired anonymity parameter.

    top_r : int, default=5
        Number of farthest candidates from which the
        representative seed is randomly selected.

    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Anonymized embedding matrix having the same shape
        as the input.

    Raises
    ------
    ValueError
        If k < 2 or k is larger than the dataset size.
    """

    if k < 2:
        raise ValueError("k must be at least 2.")

    embeddings = np.asarray(embeddings, dtype=float)

    n_samples = embeddings.shape[0]

    if k > n_samples:
        raise ValueError(
            "k cannot be larger than the dataset size."
        )

    np.random.seed(random_state)

    assigned = np.zeros(n_samples, dtype=bool)
    anonymized = np.zeros_like(embeddings)

    all_indices = np.arange(n_samples)
    clusters = []

    while np.sum(~assigned) >= 2 * k:

        remaining_indices = all_indices[~assigned]
        remaining_embeddings = embeddings[remaining_indices]

        # Compute centroid of remaining embeddings
        centroid = remaining_embeddings.mean(axis=0)

        # Randomly select one of the top-r farthest
        distances = np.linalg.norm(
            remaining_embeddings - centroid,
            axis=1,
        )

        candidate_indices = np.argsort(distances)[
            -min(top_r, len(distances)):
        ]

        first_seed = remaining_indices[
            np.random.choice(candidate_indices)
        ]

        # Randomly select one of the top-r farthest from first seed
        distances = np.linalg.norm(
            remaining_embeddings - embeddings[first_seed],
            axis=1,
        )

        candidate_indices = np.argsort(distances)[
            -min(top_r, len(distances))
        ]

        second_seed = remaining_indices[
            np.random.choice(candidate_indices)
        ]

        # Form first cluster
        distances = np.linalg.norm(
            remaining_embeddings - embeddings[first_seed],
            axis=1,
        )

        first_cluster = remaining_indices[
            np.argsort(distances)[:k]
        ]

        clusters.append(first_cluster)
        assigned[first_cluster] = True

        # Form second cluster
        remaining_indices = all_indices[~assigned]

        if len(remaining_indices) < k:
            break

        remaining_embeddings = embeddings[remaining_indices]

        distances = np.linalg.norm(
            remaining_embeddings - embeddings[second_seed],
            axis=1,
        )

        second_cluster = remaining_indices[
            np.argsort(distances)[:k]
        ]

        clusters.append(second_cluster)
        assigned[second_cluster] = True

    # Assign remaining records to the nearest existing medoid
    remaining_indices = all_indices[~assigned]

    if len(remaining_indices) > 0:

        medoids = np.array(
            [
                _compute_medoid(embeddings[cluster])
                for cluster in clusters
            ]
        )

        for index in remaining_indices:

            distances = np.linalg.norm(
                medoids - embeddings[index],
                axis=1,
            )

            nearest_cluster = np.argmin(distances)

            clusters[nearest_cluster] = np.append(
                clusters[nearest_cluster],
                index,
            )

    # Recompute medoids after merging remaining records
    medoids = np.array(
        [
            _compute_medoid(embeddings[cluster])
            for cluster in clusters
        ]
    )

    # Replace every embedding by its cluster medoid
    for cluster, medoid in zip(clusters, medoids):
        anonymized[cluster] = medoid

    return anonymized


__all__ = ["rmdav_m"]
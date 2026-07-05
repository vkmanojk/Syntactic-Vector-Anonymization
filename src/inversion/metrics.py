"""
Evaluation metrics for template inversion.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import sacrebleu
import torch

from bert_score import score as bertscore_score
from tqdm import tqdm

from .reconstruction import invert_embedding


def evaluate_embeddings(
    embeddings,
    reference_texts,
    tokenizer,
    model,
    adapter,
    device,
    *,
    label: str,
    output_dir,
):
    """
    Evaluate template inversion using BLEU and BERTScore-F1.

    Parameters
    ----------
    embeddings : numpy.ndarray or torch.Tensor
        Embedding matrix of shape (N, D).

    reference_texts : list[str]
        Ground-truth text corresponding to each embedding.

    tokenizer
        Hugging Face tokenizer.

    model
        Trained inversion model.

    adapter
        Latent expansion adapter.

    device
        Torch device.

    label : str
        Experiment label.

    output_dir : str or Path
        Directory for saving reconstruction results.

    Returns
    -------
    dict
        Dictionary containing the evaluation metrics.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Accept either NumPy arrays or PyTorch tensors
    if isinstance(embeddings, np.ndarray):
        embeddings = torch.from_numpy(
            embeddings
        ).float()

    reconstructions = []

    print(f"\nEvaluating {label}")

    for i in tqdm(
        range(len(reference_texts)),
        desc=label,
    ):

        reconstructed = invert_embedding(
            embeddings[i : i + 1],
            tokenizer,
            model,
            adapter,
            device,
        )

        reconstructions.append(reconstructed)

    reconstruction_df = pd.DataFrame(
        {
            "original": reference_texts,
            "recovered": reconstructions,
        }
    )

    reconstruction_df.to_csv(
        output_dir / "reconstruction_results.csv",
        index=False,
    )

    bleu = sacrebleu.corpus_bleu(
        reconstructions,
        [reference_texts],
    )

    MAX_WORDS = 512

    reconstructions = [
        " ".join(text.split()[:MAX_WORDS])
        for text in reconstructions
    ]

    reference_texts = [
        " ".join(text.split()[:MAX_WORDS])
        for text in reference_texts
    ]

    _, _, bert_f1 = bertscore_score(
        reconstructions,
        reference_texts,
        lang="en",
        model_type="allenai/scibert_scivocab_uncased",
        device=device,
    )

    metrics = {
        "label": label,
        "bleu": float(bleu.score),
        "bertscore_f1": float(
            bert_f1.mean().item()
        ),
    }

    pd.DataFrame([metrics]).to_csv(
        output_dir / "metrics.csv",
        index=False,
    )

    print(pd.DataFrame([metrics]))

    return metrics


__all__ = [
    "evaluate_embeddings",
]
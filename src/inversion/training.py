"""
Training utilities for the embedding inversion model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
)
from transformers.modeling_outputs import BaseModelOutput

from .model import LatentExpansionAdapter


class EmbeddingDataset(Dataset):
    """
    Dataset for embedding inversion.

    Each sample consists of

        (embedding, clinical text)

    where the embedding originates from the Google
    text-embedding-004 model.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        texts: list[str],
    ) -> None:

        if len(embeddings) != len(texts):
            raise ValueError(
                "Embeddings and texts must have the same length."
            )

        self.embeddings = torch.tensor(
            embeddings,
            dtype=torch.float32,
        )

        self.texts = list(texts)

    def __len__(self) -> int:

        return len(self.texts)

    def __getitem__(self, index):

        return (
            self.embeddings[index],
            self.texts[index],
        )


def truncate_text(
    text: str,
    *,
    max_sentences: int = 5,
) -> str:
    """
    Keep only the first few informative sentences.

    This mirrors the preprocessing used during the
    original inversion model development.
    """

    sentences = [
        s.strip()
        for s in text.split(".")
        if len(s.strip()) > 20
    ]

    return ". ".join(
        sentences[:max_sentences]
    ) + "."


def prepare_texts(
    texts: list[str],
    *,
    truncate: bool = False,
    max_sentences: int = 5,
) -> list[str]:
    """
    Prepare clinical texts for training.
    """

    if not truncate:
        return list(texts)

    return [
        truncate_text(
            text,
            max_sentences=max_sentences,
        )
        for text in texts
    ]


def freeze_scifive(
    model: T5ForConditionalGeneration,
) -> None:
    """
    Freeze all parameters except the decoder
    cross-attention layers.

    This follows the methodology adopted in the thesis.
    """

    for name, parameter in model.named_parameters():

        if "encoder" in name:

            parameter.requires_grad = False

        elif "EncDecAttention" in name:

            parameter.requires_grad = True

        else:

            parameter.requires_grad = False


def create_dataloader(
    embeddings: np.ndarray,
    texts: list[str],
    *,
    batch_size: int = 16,
    shuffle: bool = True,
) -> DataLoader:
    """
    Create a PyTorch DataLoader.
    """

    dataset = EmbeddingDataset(
        embeddings,
        texts,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

def train_epoch(
    model: T5ForConditionalGeneration,
    adapter: LatentExpansionAdapter,
    tokenizer: T5Tokenizer,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    max_length: int = 256,
) -> float:
    """
    Train the inversion model for one epoch.

    Parameters
    ----------
    model
        SciFive model.

    adapter
        Latent expansion adapter.

    tokenizer
        SciFive tokenizer.

    dataloader
        Training dataloader.

    optimizer
        Optimizer.

    device
        Torch device.

    max_length
        Maximum decoder sequence length.

    Returns
    -------
    float
        Average training loss.
    """

    model.train()
    adapter.train()

    running_loss = 0.0

    for embeddings, texts in tqdm(
        dataloader,
        leave=False,
    ):

        embeddings = embeddings.to(device)

        target = tokenizer(
            list(texts),
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        # Ignore padding tokens when computing loss
        labels = target.input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100

        optimizer.zero_grad()

        pseudo_states = adapter(embeddings)

        encoder_outputs = BaseModelOutput(
            last_hidden_state=pseudo_states,
        )

        outputs = model(
            encoder_outputs=encoder_outputs,
            labels=labels,
        )

        loss = outputs.loss

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(dataloader)

def train_inversion_model(
    embeddings: np.ndarray,
    texts: list[str],
    *,
    model_name: str = "razent/SciFive-base-PubMed",
    latent_len: int = 16,
    hidden_dim: int = 768,
    batch_size: int = 16,
    learning_rate: float = 3e-4,
    epochs: int = 20,
    max_length: int = 256,
    truncate: bool = False,
    max_sentences: int = 5,
    device: str | None = None,
):
    """
    Train the embedding inversion model.

    Parameters
    ----------
    embeddings
        Original embedding vectors.

    texts
        Corresponding clinical texts.

    Returns
    -------
    tokenizer
    model
    adapter
    """

    if device is None:

        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    device = torch.device(device)

    texts = prepare_texts(
        texts,
        truncate=truncate,
        max_sentences=max_sentences,
    )

    tokenizer = T5Tokenizer.from_pretrained(
        model_name,
    )

    model = (
        T5ForConditionalGeneration
        .from_pretrained(model_name)
        .to(device)
    )

    adapter = LatentExpansionAdapter(
        emb_dim=embeddings.shape[1],
        hidden_dim=hidden_dim,
        latent_len=latent_len,
    ).to(device)

    freeze_scifive(model)

    for parameter in adapter.parameters():

        parameter.requires_grad = True

    optimizer = torch.optim.AdamW(
        list(adapter.parameters())
        + [
            p
            for p in model.parameters()
            if p.requires_grad
        ],
        lr=learning_rate,
    )

    dataloader = create_dataloader(
        embeddings,
        texts,
        batch_size=batch_size,
        shuffle=True,
    )

    history = []

    print("\nTraining inversion model\n")

    for epoch in range(epochs):

        loss = train_epoch(
            model=model,
            adapter=adapter,
            tokenizer=tokenizer,
            dataloader=dataloader,
            optimizer=optimizer,
            device=device,
            max_length=max_length,
        )

        history.append(loss)

        print(
            f"Epoch "
            f"{epoch + 1:02d}/{epochs} | "
            f"Loss: {loss:.4f}"
        )

    return (
        tokenizer,
        model,
        adapter,
        history,
    )

from pathlib import Path

import torch

from .model import (
    load_inversion_model,
)


def save_inversion_model(
    tokenizer,
    model,
    adapter,
    output_dir,
):
    """
    Save the trained inversion model.

    Parameters
    ----------
    tokenizer
        Hugging Face tokenizer.

    model
        Trained SciFive model.

    adapter
        Trained latent expansion adapter.

    output_dir : str or Path
        Directory for saving the model.
    """

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.save_pretrained(output_dir)

    tokenizer.save_pretrained(output_dir)

    torch.save(
        adapter.state_dict(),
        output_dir / "adapter.pt",
    )

    print(
        f"\nModel saved to:\n{output_dir}"
    )


def load_or_train_inversion_model(
    embeddings,
    texts,
    model_dir,
    *,
    device=None,
    **training_kwargs,
):
    """
    Load an existing inversion model if available.
    Otherwise, train a new model and save it.

    Parameters
    ----------
    embeddings : np.ndarray
        Embedding matrix.

    texts : list[str]
        Clinical texts.

    model_dir : str or Path
        Directory containing (or receiving)
        the trained inversion model.

    device : str, optional
        Torch device.

    **training_kwargs
        Additional keyword arguments passed to
        train_inversion_model().

    Returns
    -------
    tokenizer
    model
    adapter
    device
    """

    model_dir = Path(model_dir)

    adapter_file = model_dir / "adapter.pt"

    if adapter_file.exists():

        print(
            "\nFound existing inversion model."
        )

        return load_inversion_model(
            model_dir,
            device=device,
        )

    print(
        "\nNo trained inversion model found."
    )

    print(
        "Training a new inversion model...\n"
    )

    (
        tokenizer,
        model,
        adapter,
        history,
    ) = train_inversion_model(
        embeddings,
        texts,
        device=device,
        **training_kwargs,
    )

    save_inversion_model(
        tokenizer,
        model,
        adapter,
        model_dir,
    )

    tokenizer, model, adapter, device = (
        load_inversion_model(
            model_dir,
            device=device,
        )
    )

    return (
        tokenizer,
        model,
        adapter,
        device,
    )


__all__ = [
    "EmbeddingDataset",
    "truncate_text",
    "prepare_texts",
    "freeze_scifive",
    "create_dataloader",
    "train_epoch",
    "train_inversion_model",
    "save_inversion_model",
    "load_or_train_inversion_model",
]
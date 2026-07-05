"""
Utilities for loading the trained embedding inversion model.
"""

from pathlib import Path

import torch
import torch.nn as nn

from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
)


class LatentExpansionAdapter(nn.Module):
    """
    Maps a fixed-length embedding to a sequence of latent vectors
    that can be consumed by the T5 encoder-decoder model.
    """

    def __init__(
        self,
        emb_dim: int = 768,
        hidden_dim: int = 768,
        latent_len: int = 16,
    ) -> None:

        super().__init__()

        self.latent_len = latent_len

        self.net = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2),
            nn.ReLU(),
            nn.Linear(
                emb_dim * 2,
                hidden_dim * latent_len,
            ),
        )

    def forward(
        self,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:

        batch_size = embeddings.size(0)

        latent = self.net(embeddings)

        return latent.view(
            batch_size,
            self.latent_len,
            -1,
        )


def load_inversion_model(
    model_dir,
    *,
    latent_len: int = 16,
    device: str | None = None,
):
    """
    Load the trained embedding inversion model.

    Parameters
    ----------
    model_dir : str or Path
        Directory containing the trained model.

    latent_len : int, default=16
        Length of the latent sequence.

    device : str, optional
        "cpu" or "cuda". If omitted, the best available
        device is selected automatically.

    Returns
    -------
    tokenizer
        T5 tokenizer.

    model
        Trained SciFive model.

    adapter
        Trained latent expansion adapter.

    device
        Torch device.
    """

    model_dir = Path(model_dir)

    if not model_dir.exists():
        raise FileNotFoundError(
            f"Model directory not found:\n{model_dir}"
        )

    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    tokenizer = T5Tokenizer.from_pretrained(
        model_dir
    )

    model = T5ForConditionalGeneration.from_pretrained(
        model_dir
    )

    model.to(device)
    model.eval()

    adapter = LatentExpansionAdapter(
        latent_len=latent_len
    ).to(device)

    adapter_path = model_dir / "adapter.pt"

    if not adapter_path.exists():
        raise FileNotFoundError(
            f"Adapter checkpoint not found:\n{adapter_path}"
        )

    adapter.load_state_dict(
        torch.load(
            adapter_path,
            map_location=device,
        )
    )

    adapter.eval()

    return (
        tokenizer,
        model,
        adapter,
        device,
    )


__all__ = [
    "LatentExpansionAdapter",
    "load_inversion_model",
]
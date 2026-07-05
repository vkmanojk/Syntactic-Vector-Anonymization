"""
Embedding reconstruction utilities.
"""

from __future__ import annotations

import torch

from transformers.modeling_outputs import BaseModelOutput


DEFAULT_MAX_LENGTH = 128


def encode_to_embedding(
    text: str,
    tokenizer,
    model,
    device,
    *,
    max_length: int = DEFAULT_MAX_LENGTH,
) -> torch.Tensor:
    """
    Encode a text sequence into its embedding representation using
    the encoder of the trained inversion model.

    Parameters
    ----------
    text : str
        Input text.

    tokenizer
        Hugging Face tokenizer.

    model
        Trained T5 model.

    device
        Torch device.

    max_length : int, default=128
        Maximum token length.

    Returns
    -------
    torch.Tensor
        Mean-pooled embedding.
    """

    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():

        encoder_outputs = model.encoder(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
        )

        hidden_states = encoder_outputs.last_hidden_state

        attention_mask = inputs.attention_mask.unsqueeze(-1)

        embedding = (
            hidden_states * attention_mask
        ).sum(dim=1) / attention_mask.sum(dim=1)

    return embedding


def invert_embedding(
    embedding,
    tokenizer,
    model,
    adapter,
    device,
    *,
    max_length: int = DEFAULT_MAX_LENGTH,
    num_beams: int = 4,
):
    """
    Reconstruct text from an embedding.

    Parameters
    ----------
    embedding
        Embedding tensor of shape (1, embedding_dim).

    tokenizer
        Hugging Face tokenizer.

    model
        Trained T5 model.

    adapter
        Latent expansion adapter.

    device
        Torch device.

    max_length : int, default=128
        Maximum generated sequence length.

    num_beams : int, default=4
        Beam search width.

    Returns
    -------
    str
        Reconstructed text.
    """

    if not isinstance(embedding, torch.Tensor):
        embedding = torch.tensor(
            embedding,
            dtype=torch.float32,
        )

    embedding = embedding.to(device)

    with torch.no_grad():

        pseudo_states = adapter(embedding)

        encoder_outputs = BaseModelOutput(
            last_hidden_state=pseudo_states
        )

        generated = model.generate(
            encoder_outputs=encoder_outputs,
            max_new_tokens=128,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2,
            length_penalty=0.8,
        )
        
        # generated = model.generate(
        #     encoder_outputs=encoder_outputs,
        #     max_length=max_length,
        #     num_beams=num_beams,
        #     no_repeat_ngram_size=3,
        #     repetition_penalty=1.2,
        #     length_penalty=0.8,
        #     early_stopping=True,
        # )

    return tokenizer.decode(
        generated[0],
        skip_special_tokens=True,
    )


__all__ = [
    "encode_to_embedding",
    "invert_embedding",
]
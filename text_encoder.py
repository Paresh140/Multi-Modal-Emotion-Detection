"""
RoBERTa text encoder.

CREMA-D has 12 fixed neutral sentences — no ASR needed.
The sentences add a phonetic context signal that complements audio prosody.
CLS token pooling is standard for sentence-level classification with RoBERTa.
"""
import torch
import torch.nn as nn
from transformers import AutoModel


class TextEncoder(nn.Module):
    def __init__(self, model_name: str = "roberta-base"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.embed_dim: int = self.encoder.config.hidden_size  # 768 for -base

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Returns (B, 768) CLS-token embeddings.
        """
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0, :]  # CLS

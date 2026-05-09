"""
Wav2Vec2 audio encoder with configurable layer freezing.

Wav2Vec2-base has 12 transformer layers on top of a 7-block CNN feature extractor.
Layer responsibilities (empirically established in the literature):
  - CNN + layers 0-5  : low-level acoustic features (phonemes, pitch, energy)
  - layers 6-9        : prosody and coarticulation
  - layers 10-11      : speaker-independent high-level representations

For ~5K training samples, fine-tuning all 12 layers (88M params) overfits.
Freezing the bottom 8 leaves layers 8-11 + head trainable (~18M params),
which is the right tradeoff for this data size.
"""
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model


class AudioEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "facebook/wav2vec2-base",
        freeze_feature_extractor: bool = True,
        freeze_transformer_layers: int = 8,
    ):
        super().__init__()
        
        # UPDATED: Added use_safetensors=True to bypass the CVE-2025-32434 security block
        # and ensure safe loading on modern PyTorch versions.
        self.wav2vec2 = Wav2Vec2Model.from_pretrained(
            model_name, 
            use_safetensors=True
        )
        
        self.embed_dim: int = self.wav2vec2.config.hidden_size  # 768 for -base

        # SpecAugment fires during model.train() on the CNN output features.
        self.wav2vec2.config.apply_spec_augment = True
        self.wav2vec2.config.mask_time_prob = 0.075
        self.wav2vec2.config.mask_feature_prob = 0.004

        if freeze_feature_extractor:
            self.wav2vec2.feature_extractor._freeze_parameters()

        # Freeze the bottom N transformer layers.
        # Bottom layers encode generic acoustics that don't need updating for SER.
        n_layers = len(self.wav2vec2.encoder.layers)
        for i, layer in enumerate(self.wav2vec2.encoder.layers):
            if i < freeze_transformer_layers:
                for p in layer.parameters():
                    p.requires_grad = False

        trainable = sum(p.numel() for p in self.wav2vec2.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.wav2vec2.parameters() if not p.requires_grad)
        print(
            f"AudioEncoder: {n_layers} transformer layers, "
            f"{freeze_transformer_layers} frozen | "
            f"trainable={trainable/1e6:.1f}M  frozen={frozen/1e6:.1f}M"
        )

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        """
        input_values : (B, T)  raw 16 kHz waveform
        returns      : (B, 768) mean-pooled hidden states
        """
        out = self.wav2vec2(input_values=input_values)
        return out.last_hidden_state.mean(dim=1)

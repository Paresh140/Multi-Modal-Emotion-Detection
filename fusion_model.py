"""
Multimodal fusion classifier.

Architecture
------------
Audio branch : Wav2Vec2-base → mean pool → 768-d
Text branch  : RoBERTa-base  → CLS token  → 768-d
Fusion       : concat → Linear(1536, 512) → LayerNorm → GELU → Dropout → Linear(512, 6)

When use_text=False the model reduces to an audio-only baseline with the same
head, so you can compare the two on a level playing field.
"""
import torch
import torch.nn as nn

from audio_encoder import AudioEncoder
from text_encoder import TextEncoder
from config import Config


class FusionClassifier(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.use_text = cfg.use_text

        self.audio_enc = AudioEncoder(cfg.audio_model_name, cfg.freeze_feature_extractor)
        in_dim = self.audio_enc.embed_dim  # 768

        if self.use_text:
            self.text_enc = TextEncoder(cfg.text_model_name)
            in_dim += self.text_enc.embed_dim  # 768 + 768 = 1536

        self.classifier = nn.Sequential(
            nn.Linear(in_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden_dim, cfg.num_classes),
        )

    def forward(self, batch: dict) -> torch.Tensor:
        """
        batch must contain:
          "input_values"       : (B, T)   raw waveform
          "text_input_ids"     : (B, L)   only when use_text=True
          "text_attention_mask": (B, L)   only when use_text=True
        Returns logits (B, num_classes).
        """
        audio_emb = self.audio_enc(batch["input_values"])

        if self.use_text:
            text_emb = self.text_enc(
                input_ids=batch["text_input_ids"],
                attention_mask=batch["text_attention_mask"],
            )
            fused = torch.cat([audio_emb, text_emb], dim=-1)
        else:
            fused = audio_emb

        return self.classifier(fused)

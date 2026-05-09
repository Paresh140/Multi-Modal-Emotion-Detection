from dataclasses import dataclass, field
from typing import List


@dataclass
class Config:
    # ---------- paths ----------
    data_root: str = "data/AudioWAVCremaD"
    output_dir: str = "outputs"

    # ---------- labels ----------
    emotions: List[str] = field(
        default_factory=lambda: ["angry", "disgust", "fear", "happy", "neutral", "sad"]
    )

    # ---------- audio ----------
    target_sr: int = 16000          # Wav2Vec2 native sample rate
    max_audio_seconds: float = 5.0  # clips longer than this are trimmed

    # ---------- text ----------
    use_text: bool = True   # False → audio-only baseline
    max_text_len: int = 64  # CREMA-D sentences are short (≤8 words)

    # ---------- split ----------
    val_fraction: float = 0.15
    test_fraction: float = 0.15
    seed: int = 42

    # ---------- model ----------
    audio_model_name: str = "facebook/wav2vec2-base"
    text_model_name: str = "roberta-base"
    freeze_feature_extractor: bool = True
    # Freeze the bottom N transformer layers of Wav2Vec2 (base has 12 total).
    # With 5K training samples, freezing 8 leaves ~18M trainable audio params
    # instead of 88M — roughly 290 samples/1M params vs 59 — much safer.
    # Rule of thumb: freeze_transformer_layers = total_layers - 4 for <10K samples.
    freeze_transformer_layers: int = 8
    hidden_dim: int = 512
    dropout: float = 0.4

    # ---------- training ----------
    batch_size: int = 8
    grad_accum_steps: int = 4   # effective batch = 32
    num_epochs: int = 20
    encoder_lr: float = 2e-5
    head_lr: float = 1e-4
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.1
    early_stop_patience: int = 8
    use_weighted_sampler: bool = True
    use_fp16: bool = True       # mixed-precision; set False on CPU
    # Set False on high-VRAM GPUs (≥40 GB) to avoid the slowdown from
    # recomputing activations. Keep True on T4/P100 (16 GB).
    gradient_checkpointing: bool = True

    # ---------- derived (read-only) ----------
    @property
    def num_classes(self) -> int:
        return len(self.emotions)

    @property
    def emo2id(self) -> dict:
        return {e: i for i, e in enumerate(self.emotions)}

    @property
    def id2emo(self) -> dict:
        return {i: e for i, e in enumerate(self.emotions)}

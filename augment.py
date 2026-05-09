"""
Waveform-level augmentation for raw 16 kHz audio.

Wav2Vec2 also applies its own SpecAugment internally (time & frequency
masking on the CNN feature map) when config.apply_spec_augment=True and the
model is in training mode — that covers the spectrogram domain; this module
covers the raw waveform domain.
"""
import random
import torch


def augment_waveform(wav: torch.Tensor, sr: int = 16000) -> torch.Tensor:
    # --- random gain ±15 % ---
    wav = wav * random.uniform(0.85, 1.15)

    # --- additive white Gaussian noise (~30 dB SNR) applied 50 % of the time ---
    if random.random() < 0.5:
        sigma = wav.abs().mean().item() * 0.01
        wav = wav + torch.randn_like(wav) * sigma

    # --- random time shift up to 100 ms applied 30 % of the time ---
    if random.random() < 0.3:
        max_shift = int(0.1 * sr)
        shift = random.randint(-max_shift, max_shift)
        wav = torch.roll(wav, shift)

    return wav.clamp(-1.0, 1.0)

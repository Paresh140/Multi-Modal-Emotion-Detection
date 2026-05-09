import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from augment import augment_waveform

# ---------- CREMA-D label maps ----------

CREMAD_EMOTION_MAP: Dict[str, str] = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}

# Ground-truth sentences — no Whisper needed
CREMAD_SENTENCES: Dict[str, str] = {
    "IEO": "It's eleven o'clock",
    "TIE": "That is exactly what happened",
    "IOM": "I'm on my way to the meeting",
    "IWW": "I wonder what this is about",
    "TAI": "The airplane is almost full",
    "MTI": "Maybe tomorrow it will be less crowded",
    "IWL": "I would like a new alarm clock",
    "ITH": "I think I have a doctor's appointment",
    "DFA": "Don't forget a jacket",
    "ITS": "I think I've seen this before",
    "TSI": "The surface is slick",
    "WSI": "We'll stop in a bit",
}

_FNAME_RE = re.compile(r"^(\d+)_([A-Z]+)_([A-Z]+)_([A-Z]+)\.wav$", re.IGNORECASE)


# ---------- data loading ----------

def _parse_filename(path: Path) -> Optional[Tuple[int, str, str, str]]:
    """Returns (actor_id, sentence_code, emotion_code, level) or None."""
    m = _FNAME_RE.match(path.name)
    if not m:
        return None
    return int(m.group(1)), m.group(2).upper(), m.group(3).upper(), m.group(4).upper()


def build_dataframe(data_root: str, emo2id: Dict[str, int]) -> pd.DataFrame:
    """
    Walk data_root for *.wav files, parse CREMA-D filenames, and return a
    DataFrame with columns: path, actor_id, sentence_code, emotion, label, text.
    """
    rows: List[dict] = []
    for wav_path in sorted(Path(data_root).glob("*.wav")):
        parsed = _parse_filename(wav_path)
        if parsed is None:
            continue
        actor_id, sent_code, emo_code, _ = parsed
        emotion = CREMAD_EMOTION_MAP.get(emo_code)
        if emotion is None or emotion not in emo2id:
            continue
        rows.append({
            "path": str(wav_path),
            "actor_id": actor_id,
            "sentence_code": sent_code,
            "emotion": emotion,
            "label": emo2id[emotion],
            "text": CREMAD_SENTENCES.get(sent_code, ""),
        })
    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} samples | label dist:\n{df['emotion'].value_counts().to_string()}\n")
    return df


def speaker_independent_split(
    df: pd.DataFrame,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split by actor_id so no speaker appears in more than one partition.
    This is the correct evaluation protocol for SER — avoids speaker leakage.
    """
    rng = np.random.RandomState(seed)
    actors = np.array(sorted(df["actor_id"].unique()))
    rng.shuffle(actors)

    n = len(actors)
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))

    test_actors = set(actors[:n_test])
    val_actors = set(actors[n_test: n_test + n_val])

    test_df = df[df["actor_id"].isin(test_actors)].reset_index(drop=True)
    val_df = df[df["actor_id"].isin(val_actors)].reset_index(drop=True)
    train_df = df[~df["actor_id"].isin(test_actors | val_actors)].reset_index(drop=True)

    print(
        f"Split → train={len(train_df)} "
        f"({train_df['actor_id'].nunique()} speakers) | "
        f"val={len(val_df)} ({val_df['actor_id'].nunique()} speakers) | "
        f"test={len(test_df)} ({test_df['actor_id'].nunique()} speakers)"
    )
    overlap = set(train_df.actor_id) & set(val_df.actor_id) & set(test_df.actor_id)
    assert not overlap, f"Speaker leakage detected: {overlap}"
    return train_df, val_df, test_df


# ---------- PyTorch Dataset ----------

class EmotionDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        target_sr: int = 16000,
        max_audio_seconds: float = 5.0,
        apply_augmentation: bool = False,
        use_text: bool = True,
    ):
        self.df = df.reset_index(drop=True)
        self.target_sr = target_sr
        self.max_samples = int(max_audio_seconds * target_sr)
        self.apply_augmentation = apply_augmentation
        self.use_text = use_text

    def __len__(self) -> int:
        return len(self.df)

    def _load_audio(self, path: str) -> torch.Tensor:
        wav_np, sr = sf.read(path, dtype="float32", always_2d=True)
        wav = torch.from_numpy(wav_np.mean(axis=1))  # stereo → mono
        if sr != self.target_sr:
            wav = torchaudio.functional.resample(wav, sr, self.target_sr)
        if wav.numel() > self.max_samples:
            wav = wav[: self.max_samples]
        return wav  # (T,)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        wav = self._load_audio(row["path"])
        if self.apply_augmentation:
            wav = augment_waveform(wav, self.target_sr)
        text = row["text"] if self.use_text else ""
        return {"audio": wav, "text": text, "label": int(row["label"])}


# ---------- Collator ----------

class EmotionCollator:
    """
    Pads waveforms in a batch to the same length and tokenizes text.
    When use_text=False or tokenizer is None, text tensors are omitted.
    """
    def __init__(
        self,
        tokenizer: Optional[PreTrainedTokenizerBase],
        use_text: bool = True,
        max_text_len: int = 64,
    ):
        self.tokenizer = tokenizer
        self.use_text = use_text and tokenizer is not None
        self.max_text_len = max_text_len

    def __call__(self, items: List[dict]) -> dict:
        wavs = [it["audio"] for it in items]
        max_len = max(w.shape[0] for w in wavs)
        padded = torch.stack(
            [torch.nn.functional.pad(w, (0, max_len - w.shape[0])) for w in wavs]
        )
        batch = {
            "input_values": padded,
            "labels": torch.tensor([it["label"] for it in items], dtype=torch.long),
        }
        if self.use_text:
            enc = self.tokenizer(
                [it["text"] or "" for it in items],
                padding=True,
                truncation=True,
                max_length=self.max_text_len,
                return_tensors="pt",
            )
            batch["text_input_ids"] = enc["input_ids"]
            batch["text_attention_mask"] = enc["attention_mask"]
        return batch

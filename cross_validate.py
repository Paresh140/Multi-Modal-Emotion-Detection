import torch
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from pathlib import Path
from config import Config
from dataset import build_dataframe, EmotionDataset, EmotionCollator
from fusion_model import FusionClassifier
from trainer import Trainer
from utils import set_seed, gpu_info
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, WeightedRandomSampler
import pandas as pd


def _make_weighted_sampler(train_df) -> WeightedRandomSampler:
    """Same idea as main.py: balance mini-batches by inverse class frequency."""
    num_classes = train_df["label"].nunique()
    counts = (
        train_df["label"]
        .value_counts()
        .reindex(range(num_classes), fill_value=1)
        .to_numpy(float)
    )
    weights_per_class = 1.0 / counts
    sample_weights = torch.tensor(
        [weights_per_class[l] for l in train_df["label"]], dtype=torch.float32
    )
    return WeightedRandomSampler(
        sample_weights, num_samples=len(sample_weights), replacement=True
    )


def create_datasets_from_indices(all_df, indices, cfg, apply_augmentation=True):
    """Helper to create a DataLoader from a set of indices."""
    subset_df = all_df.iloc[indices].reset_index(drop=True)
    dataset = EmotionDataset(
        subset_df,
        cfg.target_sr,
        cfg.max_audio_seconds,
        apply_augmentation=apply_augmentation,
        use_text=cfg.use_text,
    )
    return dataset


def run_fold(cfg, train_df, val_df, test_df, fold_idx, output_dir):
    """Train one fold and return validation metrics."""
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n========== Fold {fold_idx+1} ==========")

    tokenizer = AutoTokenizer.from_pretrained(cfg.text_model_name) if cfg.use_text else None
    collator = EmotionCollator(tokenizer, cfg.use_text, cfg.max_text_len)

    train_ds = EmotionDataset(
        train_df, cfg.target_sr, cfg.max_audio_seconds,
        apply_augmentation=True, use_text=cfg.use_text,
    )
    val_ds = EmotionDataset(
        val_df, cfg.target_sr, cfg.max_audio_seconds,
        apply_augmentation=False, use_text=cfg.use_text,
    )

    if getattr(cfg, "use_weighted_sampler", False):
        sampler = _make_weighted_sampler(train_df)
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            sampler=sampler,
            collate_fn=collator,
            num_workers=0,
            pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=collator,
            num_workers=0,
            pin_memory=True,
        )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size * 2, shuffle=False,
        collate_fn=collator, num_workers=0, pin_memory=True,
    )

    model = FusionClassifier(cfg).to(device)
    if cfg.gradient_checkpointing:
        model.audio_enc.wav2vec2.gradient_checkpointing_enable()
        if cfg.use_text:
            model.text_enc.encoder.gradient_checkpointing_enable()

    fold_output_dir = Path(output_dir) / f"fold_{fold_idx}"
    trainer = Trainer(model, cfg, train_df, device, fold_output_dir)
    trainer.setup_scheduler(len(train_loader))

    best_val_f1 = 0.0
    for epoch in range(1, cfg.num_epochs + 1):
        train_loss = trainer.train_epoch(train_loader)
        val_m = trainer.eval_epoch(val_loader)
        saved = trainer.maybe_save(val_m["macro_f1"])
        if saved:
            best_val_f1 = val_m["macro_f1"]
        # early stopping handled inside trainer
        if trainer.should_stop():
            break

    # Load best model and evaluate on validation set again to get final metrics
    trainer.load_best()
    final_val = trainer.eval_epoch(val_loader)
    return final_val["acc"], final_val["macro_f1"]


def cross_validate(cfg):
    # 1. Build full dataframe
    df = build_dataframe(cfg.data_root, cfg.emo2id)

    # 2. Use the same speaker‑independent split as main to separate a final test set
    # We will NOT use the validation set from that split; we'll combine train+val for CV.
    from dataset import speaker_independent_split
    train_df, val_df, test_df = speaker_independent_split(
        df, cfg.val_fraction, cfg.test_fraction, cfg.seed
    )

    # Combine train and val for cross‑validation (test remains untouched)
    cv_df = pd.concat([train_df, val_df], ignore_index=True)

    # 3. Prepare for StratifiedGroupKFold
    X = np.arange(len(cv_df))
    y = cv_df["label"].values
    # Group by speaker id to prevent leakage across folds.
    # build_dataframe() uses the CREMA-D actor id column name: "actor_id".
    groups = cv_df["actor_id"].values

    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=cfg.seed)

    fold_accuracies = []
    fold_f1s = []

    for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        train_fold_df = cv_df.iloc[train_idx].reset_index(drop=True)
        val_fold_df   = cv_df.iloc[val_idx].reset_index(drop=True)

        acc, f1 = run_fold(cfg, train_fold_df, val_fold_df, test_df, fold, cfg.output_dir)
        fold_accuracies.append(acc)
        fold_f1s.append(f1)

    # 4. Report cross‑validation results
    mean_acc = np.mean(fold_accuracies)
    std_acc  = np.std(fold_accuracies)
    mean_f1  = np.mean(fold_f1s)
    std_f1   = np.std(fold_f1s)

    print("\n" + "=" * 50)
    print("5‑Fold Cross‑Validation Results")
    print("=" * 50)
    print(f"Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"Macro F1: {mean_f1:.4f} ± {std_f1:.4f}")

    return mean_acc, mean_f1


if __name__ == "__main__":
    from config import Config

    # Use the same config as your original run
    cfg = Config(
        data_root="/content/drive/MyDrive/Multimodal_Emotion_Detection_Project/AudioWAVCremaD",
        output_dir="/content/outputs_cv",
        use_text=True,
        batch_size=8,
        grad_accum_steps=4,
        num_epochs=50,
        use_fp16=True,
        freeze_transformer_layers=8,
    )
    cross_validate(cfg)

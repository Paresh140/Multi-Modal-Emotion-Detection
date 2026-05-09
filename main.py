"""
Entry point.  Works as-is on Colab/Kaggle — just update cfg.data_root.

Usage
-----
  from main import main
  from config import Config

  cfg = Config(data_root="/content/drive/MyDrive/.../AudioWAVCremaD")
  main(cfg)
"""
from pathlib import Path

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer

from config import Config
from dataset import build_dataframe, speaker_independent_split, EmotionDataset, EmotionCollator
from fusion_model import FusionClassifier
from trainer import Trainer
from metrics import print_report, plot_confusion_matrix
from utils import set_seed, gpu_info


def _make_weighted_sampler(train_df) -> WeightedRandomSampler:
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
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def main(cfg: Config = None):
    if cfg is None:
        cfg = Config()

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    gpu_info()

    # ── data ──────────────────────────────────────────────────────────
    df = build_dataframe(cfg.data_root, cfg.emo2id)
    train_df, val_df, test_df = speaker_independent_split(
        df, cfg.val_fraction, cfg.test_fraction, cfg.seed
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.text_model_name) if cfg.use_text else None
    collator = EmotionCollator(tokenizer, cfg.use_text, cfg.max_text_len)

    train_ds = EmotionDataset(train_df, cfg.target_sr, cfg.max_audio_seconds, apply_augmentation=True,  use_text=cfg.use_text)
    val_ds   = EmotionDataset(val_df,   cfg.target_sr, cfg.max_audio_seconds, apply_augmentation=False, use_text=cfg.use_text)
    test_ds  = EmotionDataset(test_df,  cfg.target_sr, cfg.max_audio_seconds, apply_augmentation=False, use_text=cfg.use_text)

    # num_workers=0 avoids the Python 3.12 / Colab multiprocessing assert bug
    # ("can only test a child process"). The dataset is small enough that
    # single-process loading is not a throughput bottleneck.
    if cfg.use_weighted_sampler:
        sampler = _make_weighted_sampler(train_df)
        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, sampler=sampler,
            collate_fn=collator, num_workers=0, pin_memory=True,
        )
    else:
        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True,
            collate_fn=collator, num_workers=0, pin_memory=True,
        )

    val_loader = DataLoader(
        val_ds, batch_size=cfg.batch_size * 2, shuffle=False,
        collate_fn=collator, num_workers=0, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size * 2, shuffle=False,
        collate_fn=collator, num_workers=0, pin_memory=True,
    )

    # ── model ─────────────────────────────────────────────────────────
    model = FusionClassifier(cfg).to(device)

    if cfg.gradient_checkpointing:
        model.audio_enc.wav2vec2.gradient_checkpointing_enable()
        if cfg.use_text:
            model.text_enc.encoder.gradient_checkpointing_enable()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"Trainable params: {n_params:.1f} M")

    # ── training ──────────────────────────────────────────────────────
    output_dir = Path(cfg.output_dir)
    trainer = Trainer(model, cfg, train_df, device, output_dir)
    trainer.setup_scheduler(len(train_loader))

    print("\nStarting training…")
    for epoch in range(1, cfg.num_epochs + 1):
        train_loss = trainer.train_epoch(train_loader)
        val_m = trainer.eval_epoch(val_loader)

        saved = trainer.maybe_save(val_m["macro_f1"])
        tag = " ✓ best" if saved else f"  patience {trainer.patience_counter}/{cfg.early_stop_patience}"
        print(
            f"Ep {epoch:02d} | train_loss={train_loss:.4f} | "
            f"val_loss={val_m['loss']:.4f} | val_acc={val_m['acc']:.4f} | "
            f"val_mf1={val_m['macro_f1']:.4f}{tag}"
        )

        if trainer.should_stop():
            print(f"Early stopping at epoch {epoch}")
            break

    # ── test evaluation ───────────────────────────────────────────────
    print("\n=== Test Set ===")
    trainer.load_best()
    test_m = trainer.eval_epoch(test_loader)
    print(f"Test acc={test_m['acc']:.4f}  macro_f1={test_m['macro_f1']:.4f}")
    print_report(test_m["y_true"], test_m["y_pred"], cfg.emotions, "Test Report")
    plot_confusion_matrix(test_m["y_true"], test_m["y_pred"], cfg.emotions)

    return test_m


if __name__ == "__main__":
    main()

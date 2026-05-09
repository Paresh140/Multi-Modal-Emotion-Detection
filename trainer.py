"""
Training loop, optimizer, scheduler, and checkpointing.

Design decisions
----------------
* Two param groups (encoder vs head) so the pretrained encoders are updated
  with a small LR while the new head can use a larger one.
* OneCycleLR with 10 % warmup + cosine decay is state-of-the-art for
  fine-tuning transformers. ReduceLROnPlateau (reactive) consistently
  underperforms it on speech tasks.
* class_weights + label_smoothing handle the mild NEU imbalance.
* WeightedRandomSampler (built in main.py) ensures balanced mini-batches
  independent of class weights.
* Gradient checkpointing is enabled in main.py to trade compute for memory,
  allowing batch_size=8 on a T4/V100 with two transformer encoders loaded.
"""
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm.auto import tqdm
from typing import Optional

import pandas as pd
from config import Config
from metrics import compute_metrics


def _class_weights(train_df: pd.DataFrame, num_classes: int) -> torch.Tensor:
    counts = (
        train_df["label"]
        .value_counts()
        .reindex(range(num_classes), fill_value=1)
        .to_numpy(dtype=float)
    )
    w = 1.0 / counts
    return torch.tensor(w / w.mean(), dtype=torch.float32)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        cfg: Config,
        train_df: pd.DataFrame,
        device: torch.device,
        output_dir: Path,
    ):
        self.model = model
        self.cfg = cfg
        self.device = device
        self.output_dir = output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        cw = _class_weights(train_df, cfg.num_classes).to(device)
        self.loss_fn = nn.CrossEntropyLoss(weight=cw, label_smoothing=cfg.label_smoothing)

        # Separate pretrained encoder params from the new classification head.
        encoder_params, head_params = [], []
        for name, p in model.named_parameters():
            if "audio_enc.wav2vec2" in name or "text_enc.encoder" in name:
                encoder_params.append(p)
            else:
                head_params.append(p)

        self.optimizer = AdamW(
            [
                {"params": encoder_params, "lr": cfg.encoder_lr, "weight_decay": 0.01},
                {"params": head_params,    "lr": cfg.head_lr,    "weight_decay": 0.01},
            ]
        )

        # Scaler for mixed-precision — no-ops when use_fp16=False or on CPU.
        self.scaler = torch.amp.GradScaler(
            "cuda", enabled=(device.type == "cuda" and cfg.use_fp16)
        )

        self.best_mf1: float = -1.0
        self.patience_counter: int = 0
        self._scheduler: Optional[OneCycleLR] = None

    def setup_scheduler(self, steps_per_epoch: int):
        """Call after DataLoader is created so we know steps_per_epoch."""
        effective_steps = (
            self.cfg.num_epochs * steps_per_epoch // self.cfg.grad_accum_steps
        )
        self._scheduler = OneCycleLR(
            self.optimizer,
            max_lr=[self.cfg.encoder_lr, self.cfg.head_lr],
            total_steps=effective_steps,
            pct_start=0.1,          # 10 % warmup
            anneal_strategy="cos",
        )

    # ------------------------------------------------------------------

    def train_epoch(self, loader) -> float:
        self.model.train()
        losses: list = []
        self.optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(tqdm(loader, desc="  train", leave=False)):
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }

            with torch.amp.autocast("cuda", enabled=self.scaler.is_enabled()):
                logits = self.model(batch)
                loss = self.loss_fn(logits, batch["labels"]) / self.cfg.grad_accum_steps

            self.scaler.scale(loss).backward()

            if (step + 1) % self.cfg.grad_accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                if self._scheduler is not None:
                    self._scheduler.step()

            losses.append(loss.item() * self.cfg.grad_accum_steps)

        return float(np.mean(losses))

    @torch.no_grad()
    def eval_epoch(self, loader) -> dict:
        self.model.eval()
        all_pred, all_true = [], []
        total_loss = 0.0
        n = 0

        for batch in tqdm(loader, desc="  eval", leave=False):
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            logits = self.model(batch)
            loss = self.loss_fn(logits, batch["labels"])
            total_loss += loss.item() * batch["labels"].size(0)
            n += batch["labels"].size(0)
            all_pred.append(logits.argmax(-1).cpu().numpy())
            all_true.append(batch["labels"].cpu().numpy())

        y_pred = np.concatenate(all_pred)
        y_true = np.concatenate(all_true)
        return compute_metrics(y_true, y_pred, total_loss / max(1, n))

    # ------------------------------------------------------------------

    def maybe_save(self, mf1: float, tag: str = "best") -> bool:
        if mf1 > self.best_mf1:
            self.best_mf1 = mf1
            torch.save(self.model.state_dict(), self.output_dir / f"{tag}_model.pt")
            self.patience_counter = 0
            return True
        self.patience_counter += 1
        return False

    def should_stop(self) -> bool:
        return self.patience_counter >= self.cfg.early_stop_patience

    def load_best(self, tag: str = "best"):
        path = self.output_dir / f"{tag}_model.pt"
        self.model.load_state_dict(torch.load(path, map_location=self.device))

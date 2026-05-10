from __future__ import annotations

import argparse
import copy
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from src.data import (
    HindiMNISTTrain,
    build_eval_transform,
    build_train_transform,
    load_fold_split,
    mixup_cutmix_collate,
)

ROOT = Path(__file__).resolve().parent.parent


class ModelEMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(msd[k].detach(), alpha=1 - self.decay)
            else:
                v.copy_(msd[k])


def soft_ce(logits: torch.Tensor, soft_targets: torch.Tensor,
            label_smoothing: float = 0.0) -> torch.Tensor:
    if label_smoothing > 0:
        n = soft_targets.size(-1)
        soft_targets = soft_targets * (1 - label_smoothing) + label_smoothing / n
    log_probs = F.log_softmax(logits, dim=-1)
    return -(soft_targets * log_probs).sum(dim=-1).mean()


def cosine_lr(step: int, total_steps: int, warmup_steps: int, base_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * base_lr * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def evaluate(model, loader, device) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    all_probs, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(imgs)
        probs = F.softmax(logits.float(), dim=-1).cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy() if torch.is_tensor(labels) else np.asarray(labels))
    probs = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    acc = (probs.argmax(axis=1) == labels).mean()
    return float(acc), probs, labels


def train_one_fold(cfg: dict, fold: int, debug: bool = False,
                   pseudo_csv: Path | None = None) -> dict:
    arch = cfg["arch"]
    imgsz = cfg["imgsz"]
    epochs = 2 if debug else cfg["epochs"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_dir = ROOT / "checkpoints" / arch / f"fold{fold}"
    oof_dir = ROOT / "oof"
    log_dir = ROOT / "logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    oof_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    train_df, val_df = load_fold_split(fold)
    if pseudo_csv is not None and Path(pseudo_csv).exists():
        pseudo = pd.read_csv(pseudo_csv)
        pseudo["pseudo"] = True
        if "fold" not in pseudo.columns:
            pseudo["fold"] = -1
        train_df["pseudo"] = False
        train_df = pd.concat([train_df, pseudo[["Id", "Category", "fold", "pseudo"]]],
                             ignore_index=True)
        print(f"[fold {fold}] added {len(pseudo)} pseudo-labeled rows")

    train_ds = HindiMNISTTrain(train_df, transform=build_train_transform(imgsz))
    val_ds = HindiMNISTTrain(val_df, transform=build_eval_transform(imgsz))

    def collate_train(batch):
        return mixup_cutmix_collate(
            batch,
            num_classes=10,
            mixup_alpha=cfg["mixup_alpha"],
            cutmix_alpha=cfg["cutmix_alpha"],
            prob=cfg["mixup_prob"],
        )

    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["num_workers"], pin_memory=True, drop_last=True,
        collate_fn=collate_train, persistent_workers=cfg["num_workers"] > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], pin_memory=True,
        persistent_workers=cfg["num_workers"] > 0,
    )

    model = timm.create_model(arch, pretrained=cfg["pretrained"], num_classes=10).to(device)
    ema = ModelEMA(model, decay=cfg["ema_decay"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["amp"] and device.type == "cuda")

    total_steps = epochs * len(train_loader)
    warmup_steps = cfg["warmup_epochs"] * len(train_loader)
    step = 0
    best_acc = -1.0
    best_probs = None
    best_labels = None
    log_rows = []

    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        n_batches = 0
        for imgs, soft_targets in train_loader:
            lr = cosine_lr(step, total_steps, warmup_steps, cfg["lr"])
            for g in opt.param_groups:
                g["lr"] = lr

            imgs = imgs.to(device, non_blocking=True)
            soft_targets = soft_targets.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=cfg["amp"] and device.type == "cuda"):
                logits = model(imgs)
                loss = soft_ce(logits, soft_targets, label_smoothing=cfg["label_smoothing"])

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            ema.update(model)

            running_loss += float(loss.item())
            n_batches += 1
            step += 1

        train_loss = running_loss / max(1, n_batches)
        val_acc, val_probs, val_labels = evaluate(ema.module, val_loader, device)
        elapsed = time.time() - t0
        print(f"[{arch} fold{fold}] epoch {epoch+1}/{epochs} "
              f"loss={train_loss:.4f} val_acc={val_acc:.4f} lr={lr:.2e} ({elapsed:.1f}s)")
        log_rows.append({"epoch": epoch + 1, "train_loss": train_loss,
                         "val_acc": val_acc, "lr": lr, "time_s": elapsed})

        if val_acc > best_acc:
            best_acc = val_acc
            best_probs = val_probs
            best_labels = val_labels
            torch.save({
                "ema_state_dict": ema.module.state_dict(),
                "model_state_dict": model.state_dict(),
                "cfg": cfg,
                "fold": fold,
                "val_acc": val_acc,
                "epoch": epoch + 1,
            }, ckpt_dir / "best.pt")

    pd.DataFrame(log_rows).to_csv(log_dir / f"{arch}_fold{fold}.csv", index=False)
    val_ids_in_order = val_df["Id"].to_numpy()
    np.save(oof_dir / f"{arch}_fold{fold}.npy", best_probs)
    np.save(oof_dir / f"{arch}_fold{fold}_ids.npy", val_ids_in_order)
    np.save(oof_dir / f"{arch}_fold{fold}_labels.npy", best_labels)
    return {"fold": fold, "best_val_acc": best_acc}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--debug", action="store_true",
                    help="Run only 2 epochs for smoke testing")
    ap.add_argument("--pseudo", type=Path, default=None,
                    help="Optional path to pseudo-label CSV (Id,Category)")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    train_one_fold(cfg, args.fold, debug=args.debug, pseudo_csv=args.pseudo)


if __name__ == "__main__":
    main()

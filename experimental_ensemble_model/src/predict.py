from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from src.data import HindiMNISTTest, build_tta_transforms

ROOT = Path(__file__).resolve().parent.parent


@torch.no_grad()
def predict_with_tta(model, test_ids, transforms, device, batch_size=256, num_workers=4):
    summed = None
    for t in transforms:
        ds = HindiMNISTTest(test_ids, transform=t)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
        chunks = []
        for imgs, _ in loader:
            imgs = imgs.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(imgs)
            chunks.append(F.softmax(logits.float(), dim=-1).cpu().numpy())
        probs = np.concatenate(chunks, axis=0)
        summed = probs if summed is None else summed + probs
    return summed / len(transforms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--no-tta", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    arch = cfg["arch"]
    imgsz = cfg["imgsz"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt_path = ROOT / "checkpoints" / arch / f"fold{args.fold}" / "best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = timm.create_model(arch, pretrained=False, num_classes=10).to(device)
    model.load_state_dict(ckpt["ema_state_dict"])
    model.eval()

    test_csv = pd.read_csv(ROOT / "data" / "test.csv")
    test_ids = test_csv["Id"].astype(int).tolist()

    transforms = build_tta_transforms(imgsz) if not args.no_tta else build_tta_transforms(imgsz)[:1]
    probs = predict_with_tta(model, test_ids, transforms, device,
                             batch_size=cfg["batch_size"], num_workers=cfg["num_workers"])

    out_dir = ROOT / "preds"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{arch}_fold{args.fold}.npy", probs)
    if not (out_dir / "test_ids.npy").exists():
        np.save(out_dir / "test_ids.npy", np.asarray(test_ids, dtype=np.int64))


if __name__ == "__main__":
    main()

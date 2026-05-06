"""Dataset and augmentation pipeline for Hindi MNIST."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import albumentations as A
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "data" / "train" / "train"
TEST_DIR = ROOT / "data" / "test" / "test"


def _gray_to_rgb_normalize() -> A.Compose:
    return A.Compose([
        A.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=255.0),
        ToTensorV2(),
    ])


def build_train_transform(imgsz: int = 64) -> A.Compose:
    """Online augmentations tuned for handwritten Devanagari digits.
    No flips — would change digit identity (e.g. ६ vs ९)."""
    return A.Compose([
        A.Resize(imgsz, imgsz),
        A.Affine(
            rotate=(-10, 10),
            scale=(0.85, 1.15),
            translate_percent=(-0.10, 0.10),
            shear=(-5, 5),
            p=0.8,
        ),
        A.ElasticTransform(alpha=30.0, sigma=4.0, p=0.5),
        A.GridDistortion(num_steps=5, distort_limit=0.2, p=0.3),
        A.CoarseDropout(
            num_holes_range=(1, 2),
            hole_height_range=(6, 10),
            hole_width_range=(6, 10),
            fill=0,
            p=0.5,
        ),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        A.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=255.0),
        ToTensorV2(),
    ])


def build_eval_transform(imgsz: int = 64) -> A.Compose:
    return A.Compose([
        A.Resize(imgsz, imgsz),
        A.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=255.0),
        ToTensorV2(),
    ])


def build_tta_transforms(imgsz: int = 64) -> list[A.Compose]:
    """11-view TTA: original + 6 rotations + 2 translates + 2 scales."""
    base = [A.Resize(imgsz, imgsz)]
    norm = [A.Normalize(mean=(0.5,), std=(0.5,), max_pixel_value=255.0), ToTensorV2()]

    def make(ops):
        return A.Compose(base + ops + norm)

    return [
        make([]),
        make([A.Affine(rotate=(-15, -15), p=1.0)]),
        make([A.Affine(rotate=(-10, -10), p=1.0)]),
        make([A.Affine(rotate=(-5, -5), p=1.0)]),
        make([A.Affine(rotate=(5, 5), p=1.0)]),
        make([A.Affine(rotate=(10, 10), p=1.0)]),
        make([A.Affine(rotate=(15, 15), p=1.0)]),
        make([A.Affine(translate_percent=(0.05, 0.05), p=1.0)]),
        make([A.Affine(translate_percent=(-0.05, -0.05), p=1.0)]),
        make([A.Affine(scale=(1.1, 1.1), p=1.0)]),
        make([A.Affine(scale=(0.9, 0.9), p=1.0)]),
    ]


class HindiMNISTTrain(Dataset):
    """Reads images from data/train/train/<class>/<id>.png using folds.csv.

    Includes optional pseudo-labeled test images (data/test/test/<id>.png).
    Pseudo rows are flagged in the dataframe with column `pseudo=True`.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        transform: A.Compose,
        train_dir: Path = TRAIN_DIR,
        test_dir: Path = TEST_DIR,
    ):
        required = {"Id", "Category"}
        assert required.issubset(df.columns), df.columns
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.has_pseudo = "pseudo" in self.df.columns

    def __len__(self) -> int:
        return len(self.df)

    def _path_for(self, row) -> Path:
        if self.has_pseudo and bool(row["pseudo"]):
            return self.test_dir / f"{int(row['Id'])}.png"
        return self.train_dir / str(int(row["Category"])) / f"{int(row['Id'])}.png"

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = self._path_for(row)
        img = np.array(Image.open(path).convert("L"))
        img = np.expand_dims(img, axis=-1)  # HxWx1
        img = self.transform(image=img)["image"]
        # Replicate gray channel to 3 for ImageNet-pretrained backbones.
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        label = int(row["Category"])
        return img, label


class HindiMNISTTest(Dataset):
    """Test set — flat data/test/test/<id>.png."""

    def __init__(self, ids: list[int], transform: A.Compose, test_dir: Path = TEST_DIR):
        self.ids = list(ids)
        self.transform = transform
        self.test_dir = test_dir

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        img_id = self.ids[idx]
        img = np.array(Image.open(self.test_dir / f"{img_id}.png").convert("L"))
        img = np.expand_dims(img, axis=-1)
        img = self.transform(image=img)["image"]
        if img.shape[0] == 1:
            img = img.repeat(3, 1, 1)
        return img, img_id


def load_fold_split(fold: int, folds_csv: Path = ROOT / "data" / "folds.csv"):
    df = pd.read_csv(folds_csv)
    train_df = df[df["fold"] != fold].copy()
    val_df = df[df["fold"] == fold].copy()
    return train_df, val_df


def mixup_cutmix_collate(batch, num_classes: int = 10,
                         mixup_alpha: float = 0.2, cutmix_alpha: float = 1.0,
                         prob: float = 0.5):
    """Collate that randomly applies either mixup or cutmix (50/50) with given probability."""
    imgs = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    onehot = torch.nn.functional.one_hot(labels, num_classes).float()

    if torch.rand(1).item() > prob:
        return imgs, onehot

    use_cutmix = torch.rand(1).item() < 0.5
    perm = torch.randperm(imgs.size(0))

    if use_cutmix:
        lam = float(np.random.beta(cutmix_alpha, cutmix_alpha))
        _, _, h, w = imgs.shape
        cut_h, cut_w = int(h * (1 - lam) ** 0.5), int(w * (1 - lam) ** 0.5)
        cy, cx = np.random.randint(h), np.random.randint(w)
        y1, y2 = max(0, cy - cut_h // 2), min(h, cy + cut_h // 2)
        x1, x2 = max(0, cx - cut_w // 2), min(w, cx + cut_w // 2)
        imgs[:, :, y1:y2, x1:x2] = imgs[perm, :, y1:y2, x1:x2]
        lam = 1 - ((y2 - y1) * (x2 - x1) / (h * w))
    else:
        lam = float(np.random.beta(mixup_alpha, mixup_alpha))
        imgs = lam * imgs + (1 - lam) * imgs[perm]

    targets = lam * onehot + (1 - lam) * onehot[perm]
    return imgs, targets

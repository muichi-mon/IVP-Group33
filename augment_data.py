"""
Offline data augmentation for Hindi MNIST.
Generates augmented copies of each training image to disk so YOLO can use them.

Augmentations applied (lecture-recommended for handwritten digits):
- Elastic distortions (VERY powerful for digits)
- Gaussian noise
- Stroke thickness variation (morphological dilation/erosion)
- Small random shifts (YOLO also does this at train time)

Avoided: flips, extreme transforms.

Usage:
    python augment_data.py              # generates 2 aug copies per image
    python augment_data.py --copies 3   # generate 3 copies per image
    python augment_data.py --workers 8  # override CPU count
    python augment_data.py --clean      # delete previously generated aug files
"""

import argparse
import os
import time
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates, grey_dilation, grey_erosion

ROOT = Path(__file__).parent
TRAIN_SRC = ROOT / "data" / "train" / "train"
AUG_SUFFIX = "_aug"


def elastic_transform(img: np.ndarray, alpha: float, sigma: float,
                      rng: np.random.Generator) -> np.ndarray:
    shape = img.shape
    dx = gaussian_filter(rng.uniform(-1, 1, shape), sigma) * alpha
    dy = gaussian_filter(rng.uniform(-1, 1, shape), sigma) * alpha
    y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    indices = np.clip(y + dy, 0, shape[0] - 1), np.clip(x + dx, 0, shape[1] - 1)
    return map_coordinates(img, indices, order=1, mode="reflect").reshape(shape)


def stroke_thickness(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    choice = rng.integers(0, 3)
    if choice == 0:
        return grey_dilation(img, size=(2, 2))
    if choice == 1:
        return grey_erosion(img, size=(2, 2))
    return img


def gaussian_noise(img: np.ndarray, rng: np.random.Generator, std: float) -> np.ndarray:
    noise = rng.normal(0, std, img.shape)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def random_shift(img: np.ndarray, rng: np.random.Generator, max_shift: int = 2) -> np.ndarray:
    dy, dx = rng.integers(-max_shift, max_shift + 1, size=2)
    return np.roll(img, shift=(dy, dx), axis=(0, 1))


def augment_one(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = elastic_transform(img.astype(np.float32),
                            alpha=rng.uniform(20, 40),
                            sigma=rng.uniform(3, 5),
                            rng=rng).astype(np.uint8)
    out = stroke_thickness(out, rng)
    out = gaussian_noise(out, rng, std=rng.uniform(4, 10))
    out = random_shift(out, rng, max_shift=2)
    return out


def process_image(task):
    """Worker: augment one source image into `copies` output files. Skips existing outputs."""
    img_path, copies, seed = task
    class_dir = img_path.parent
    stem = img_path.stem

    # Skip if all copies already exist (resumable)
    pending = [i for i in range(copies)
               if not (class_dir / f"{stem}{AUG_SUFFIX}{i}.png").exists()]
    if not pending:
        return 0

    rng = np.random.default_rng(seed)
    img = np.array(Image.open(img_path).convert("L"))

    written = 0
    for i in pending:
        aug = augment_one(img, rng)
        Image.fromarray(aug).save(class_dir / f"{stem}{AUG_SUFFIX}{i}.png")
        written += 1
    return written


def clean_augmented(train_dir: Path) -> int:
    removed = 0
    for p in train_dir.rglob(f"*{AUG_SUFFIX}*.png"):
        p.unlink()
        removed += 1
    return removed


def collect_tasks(copies: int):
    tasks = []
    for class_dir in sorted(TRAIN_SRC.iterdir()):
        if not class_dir.is_dir():
            continue
        for p in class_dir.glob("*.png"):
            if AUG_SUFFIX in p.stem:
                continue
            # Deterministic seed per image from hash of relative path
            seed = abs(hash(str(p.relative_to(TRAIN_SRC)))) % (2**32)
            tasks.append((p, copies, seed))
    return tasks


def run(copies: int, workers: int):
    tasks = collect_tasks(copies)
    total_originals = len(tasks)
    print(f"Found {total_originals} originals. Generating {copies} copies each "
          f"using {workers} workers...")

    t0 = time.time()
    total_written = 0
    done = 0

    with Pool(processes=workers) as pool:
        for written in pool.imap_unordered(process_image, tasks, chunksize=32):
            total_written += written
            done += 1
            if done % 500 == 0 or done == total_originals:
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total_originals - done) / rate if rate > 0 else 0
                print(f"  {done}/{total_originals} originals processed "
                      f"({rate:.0f} img/s, ETA {eta:.0f}s, "
                      f"{total_written} files written)")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s. "
          f"Wrote {total_written} augmented images "
          f"(skipped {total_originals * copies - total_written} already-existing).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--copies", type=int, default=2,
                        help="augmented copies per original (default 2)")
    parser.add_argument("--workers", type=int, default=max(1, cpu_count() - 1),
                        help="parallel worker processes (default: cpu_count - 1)")
    parser.add_argument("--clean", action="store_true",
                        help="delete previously generated augmented files and exit")
    args = parser.parse_args()

    if args.clean:
        n = clean_augmented(TRAIN_SRC)
        print(f"Removed {n} augmented files.")
    else:
        run(args.copies, args.workers)

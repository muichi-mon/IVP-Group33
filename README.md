# IVP-Group33 — Hindi MNIST Classification

Devanagari digit (0–9) classification for the IIVP-2026 Kaggle challenge.

## Group 33

1. Jan Nguyen
2. NM Husar — i6329502
3. Rajveer Jadhav — i6371243

## Pipeline

3 timm architectures × 5-fold stratified CV → 7-view TTA → softmax ensemble → optional pseudo-label round.

| Stage | Tool |
|---|---|
| Fold split | `src/folds.py` (StratifiedKFold on `data/train.csv`) |
| Augmentation | `src/data.py` (albumentations: affine + elastic + grid distortion + coarse dropout + mixup/cutmix) |
| Training | `src/train.py` + `src/run_kfold.py` (AdamW, cosine LR, AMP, EMA, label smoothing) |
| Inference | `src/predict.py` (TTA: orig + 4 rotations + 2 translates) |
| Ensembling | `src/ensemble.py` (mean softmax across all `preds/*.npy`) |
| Pseudo-labels | `src/pseudo_label.py` (threshold ≥ 0.99 max-prob) |

Architectures (in `configs/`): `convnext_tiny`, `tf_efficientnetv2_s`, `resnet50d`. Image size 64.

## Reproducing the submission

```bash
make install            # pip install -r requirements.txt
make clean-aug          # remove offline-augmented PNGs (one-time)
make folds              # writes data/folds.csv
make train-all          # 15 models — ~2h on a single 4090
make predict-all        # TTA inference, writes preds/*.npy
make ensemble           # writes submission.csv

# Optional pseudo-label round (~30 min more)
make pseudo-round       # writes submission_pseudo.csv
```

Quick sanity check (2 epochs, one fold):

```bash
make smoke
```

## Layout

```
configs/                 per-arch hyperparameters
src/                     training + inference + ensembling
data/
  train.csv  test.csv  sample_submission.csv  folds.csv
  train/train/<class>/<id>.png
  test/test/<id>.png
checkpoints/{arch}/fold{k}/best.pt    saved per fold
oof/{arch}_fold{k}.npy                out-of-fold softmax
preds/{arch}_fold{k}.npy              TTA test softmax
legacy_yolo/                          original YOLO baseline (kept for writeup)
```

## Legacy YOLO baseline

The original Ultralytics `yolo26m-cls` baseline lives in `legacy_yolo/` for reproducibility comparison. Its training script and weights have been preserved unchanged.

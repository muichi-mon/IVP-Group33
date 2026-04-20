from pathlib import Path
import pandas as pd
import torch
from ultralytics import YOLO

ROOT = Path(__file__).parent
DATA = ROOT / "data"
TRAIN_SRC = DATA / "train" / "train"
TEST_SRC = DATA / "test" / "test"
WEIGHTS = ROOT / "yolo26m-cls.pt"


if __name__ == "__main__":
    assert torch.cuda.is_available(), "No GPU detected — training on CPU would be extremely slow."
    gpu_name = torch.cuda.get_device_name(0)
    vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name}  |  VRAM: {vram:.1f} GB")

    model = YOLO(str(WEIGHTS))

    model.train(
        data=str(TRAIN_SRC),
        epochs=50,
        imgsz=224,
        batch=-1,
        device=0,
        workers=8,
        cache="disk",
        amp=True,
        cos_lr=True,
        patience=15,
        degrees=10,
        fliplr=0.0,
        hsv_s=0.0,
        hsv_h=0.0,
        mosaic=0.0,
    )

    best = Path("runs/classify/train/weights/best.pt")
    if best.exists():
        model = YOLO(str(best))

    results = model.predict(
        source=str(TEST_SRC),
        save=False,
        batch=256,
        device=0,
        workers=8,
        verbose=False,
    )
    pred_map = {Path(r.path).stem: int(r.probs.top1) for r in results}

    test_csv = pd.read_csv(DATA / "test.csv")
    test_csv["Category"] = test_csv["Id"].astype(str).map(pred_map)
    test_csv.to_csv(ROOT / "submission.csv", index=False)
    print(f"submission.csv saved ({len(test_csv)} rows)")

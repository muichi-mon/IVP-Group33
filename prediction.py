from pathlib import Path
import pandas as pd
from ultralytics import YOLO

ROOT= Path(r"c:\Users\juras\OneDrive\Počítač\maastricht\kaggle-iivp\IVP-Group33")
DATA= ROOT / "data"
TEST_SRC= DATA / "test" / "test"

model = YOLO(ROOT / "runs" / "classify" / "train" / "weights" / "best.pt")
results = model.predict(source=str(TEST_SRC), save=False)

pred_map = {Path(r.path).stem: int(r.probs.top1) for r in results}

test_csv = pd.read_csv(DATA / "test.csv")
test_csv["Category"] = test_csv["Id"].astype(str).map(pred_map)
test_csv.to_csv(ROOT / "submission.csv", index=False)
print(f"submission.csv saved ({len(test_csv)} rows)")
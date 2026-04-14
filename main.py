from pathlib import Path
import pandas as pd
from ultralytics import YOLO

ROOT= Path(__file__).parent
DATA = ROOT / "data"
TRAIN_SRC = DATA / "train" / "train"
TEST_SRC = DATA / "test"  / "test"

if __name__ == "__main__":
    model = YOLO("IVP-Group33\\yolo26m-cls.pt")
    model.train(data=str(TRAIN_SRC), epochs=40)

    results = model.predict(source=str(TEST_SRC), save=False)
    pred_map = {Path(r.path).stem: int(r.probs.top1) for r in results}

    test_csv = pd.read_csv(DATA / "test.csv")
    test_csv["Category"] = test_csv["Id"].astype(str).map(pred_map)
    test_csv.to_csv(ROOT / "submission.csv", index=False)
    print(f"submission.csv saved ({len(test_csv)} rows)")
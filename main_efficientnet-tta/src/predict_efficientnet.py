from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from torchvision.models import efficientnet_b0


ROOT = Path(__file__).parent
DATA = ROOT / "data"
TEST_DIR = DATA / "test" / "test"
MODEL_PATH = ROOT / "best_efficientnet.pt"
NUM_CLASSES = 10
BATCH_SIZE = 64
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONF_THRESH = 0.75


base_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

def preprocess(img):
    return img.convert("L").convert("RGB")

def to_tensor(img):
    return base_transform(img)

def _rotate(img, deg):
    return img.rotate(deg, resample=Image.BICUBIC, expand=False)

def _crop_top(img, frac=0.55):
    w, h = img.size
    return img.crop((0, 0, w, int(h * frac))).resize((w, h), Image.LANCZOS)

def _crop_bottom(img, frac=0.55):
    w, h = img.size
    return img.crop((0, int(h * (1 - frac)), w, h)).resize((w, h), Image.LANCZOS)

def _crop_middle(img, top=0.25, bot=0.75):
    w, h = img.size
    return img.crop((0, int(h * top), w, int(h * bot))).resize((w, h), Image.LANCZOS)

def _crop_center(img, frac=0.75):
    w, h = img.size
    mx, my = int(w * (1 - frac) / 2), int(h * (1 - frac) / 2)
    return img.crop((mx, my, w - mx, h - my)).resize((w, h), Image.LANCZOS)

# Crop to the sub-region that best separates each Hindi digit from its
# nearest look-alike (3<->9, 0<->1, 5<->6, 7<->8).
_DIGIT_CROP = {
    0: lambda img: _crop_center(img, 0.80),   # spiral loop is the whole shape
    1: lambda img: _crop_top(img, 0.55),       # hook at top-right is key
    2: lambda img: _crop_top(img, 0.55),       # loop at top is key
    3: lambda img: _crop_middle(img, 0.20, 0.75),  # zigzag lives in the middle
    4: lambda img: _crop_center(img, 0.75),    # crossing loops are centered
    5: lambda img: _crop_bottom(img, 0.55),    # Y-fork is in the lower half
    6: lambda img: _crop_top(img, 0.55),       # double bump is at the top
    7: lambda img: _crop_bottom(img, 0.55),    # open-U curve is at the bottom
    8: lambda img: _crop_top(img, 0.55),       # horizontal bar + arc at top
    9: lambda img: _crop_middle(img, 0.25, 0.80),  # S-curve middle (vs digit 3)
}

def tta_views(img, digit):
    img = preprocess(img)

    if digit == 9:
        # emboss is the only view that reliably votes for Hindi 9
        return [
            to_tensor(img),
            to_tensor(Image.blend(img, img.filter(ImageFilter.EMBOSS), 0.7)),
        ]

    crop = _DIGIT_CROP[digit](img)
    sharp = ImageEnhance.Sharpness(img).enhance(2.5)

    if digit in [3, 5, 6, 7]:
        # tilt matters for zigzag (3), fork/U-curve (5,7), double-bump (6)
        return [
            to_tensor(img),
            to_tensor(crop),
            to_tensor(_rotate(img, 10)),
            to_tensor(_rotate(img, -10)),
        ]

    if digit == 4:
        emb = to_tensor(Image.blend(img, img.filter(ImageFilter.EMBOSS), 0.7))
        return [
            to_tensor(img),
            to_tensor(_rotate(img, 10)),
            to_tensor(_rotate(img, -10)),
            emb,
            emb,
        ]

    # digits 0,1,2,8 — crop + sharpen to expose distinctive edges
    return [
        to_tensor(img),
        to_tensor(crop),
        to_tensor(sharp),
    ]

class TestDataset(Dataset):
    def __init__(self, folder):
        self.paths = sorted(folder.iterdir())

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        return base_transform(img), path.stem


def run_tta(model, path, digit):
    img = Image.open(path)
    probs_sum = torch.zeros(NUM_CLASSES)
    views = tta_views(img, digit)

    with torch.no_grad():
        for v in views:
            out = model(v.unsqueeze(0).to(DEVICE))
            probs = F.softmax(out, dim=1).cpu().squeeze()
            probs_sum += probs

    probs_avg = probs_sum / len(views)
    return int(probs_avg.argmax()), float(probs_avg.max())

if __name__ == "__main__":
    print(f"Device: {DEVICE}")

    model = efficientnet_b0(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    dataset = TestDataset(TEST_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    rows = []

    with torch.no_grad():
        for images, stems in tqdm(loader, desc="baseline"):
            images = images.to(DEVICE)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)

            for stem, prob in zip(stems, probs.cpu()):
                rows.append({
                    "Id": stem,
                    "Category": int(prob.argmax()),
                    "Confidence": float(prob.max())
                })

    # apply TTA for uncertain predictions
    path_map = {p.stem: p for p in TEST_DIR.iterdir()}

    tta_used = 0
    tta_changed = 0

    for row in tqdm(rows, desc="TTA"):
        if row["Confidence"] < CONF_THRESH:
            old = row["Category"]

            new, conf = run_tta(model, path_map[row["Id"]], old)

            row["Category"] = new
            row["Confidence"] = conf

            tta_used += 1
            if new != old:
                tta_changed += 1

    print(f"TTA applied: {tta_used}, changed: {tta_changed}")

    test_csv = pd.read_csv(DATA / "test.csv")

    pred_map = {r["Id"]: r["Category"] for r in rows}
    test_csv["Category"] = test_csv["Id"].astype(str).map(pred_map)

    out_path = ROOT / "submission_efficientnet.csv"
    test_csv[["Id", "Category"]].to_csv(out_path, index=False)

    print(f"Saved: {out_path}")
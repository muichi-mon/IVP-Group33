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

#apply TTA for specific digits according what they need
def tta_views(img, digit):
    img = preprocess(img)

    views = []
    views.append(to_tensor(img))

    views.append(to_tensor(ImageOps.autocontrast(img)))
    views.append(to_tensor(ImageEnhance.Contrast(img).enhance(1.5)))

    # digit-specific tweaks
    if digit == 4:
        emb = Image.blend(img, img.filter(ImageFilter.EMBOSS), 0.7)
        return [
            to_tensor(img),
            to_tensor(emb)
        ]

    if digit == 9:
        views.append(to_tensor(Image.blend(img, img.filter(ImageFilter.EMBOSS), 0.7)))
        w, h = img.size
        top = img.crop((0, 0, w, int(h * 0.6))).resize((224, 224))
        views.append(to_tensor(top))
        views.append(to_tensor(ImageEnhance.Sharpness(img).enhance(3.0)))
        
    if digit in [2, 3, 5]:
        # bottom-heavy digits
        w, h = img.size
        crop = img.crop((0, int(h*0.4), w, h))
        crop = crop.resize((224, 224))
        views.append(to_tensor(crop))

    if digit in [1, 7]:
        # top-heavy digits
        w, h = img.size
        crop = img.crop((0, 0, w, int(h*0.6)))
        crop = crop.resize((224, 224))
        views.append(to_tensor(crop))

    if digit in [8]:
        views.append(to_tensor(ImageOps.equalize(img)))

    return views

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
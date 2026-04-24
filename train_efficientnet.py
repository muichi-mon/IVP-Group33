from pathlib import Path
import subprocess
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

ROOT = Path(__file__).parent
DATA = ROOT / "data"
TRAIN_DIR = DATA / "train" / "train_split" / "train"
VAL_DIR   = DATA / "train" / "train_split" / "val"
BATCH_SIZE = 64
EPOCHS     = 20
LR         = 5e-4
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = ROOT / "best_efficientnet.pt"

# augmentations (similar like in yolo)
train_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
    transforms.RandomApply([
        transforms.GaussianBlur(kernel_size=3)
    ], p=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

val_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])


if __name__ == "__main__":

    train_dataset = datasets.ImageFolder(str(TRAIN_DIR), transform=train_transforms)
    val_dataset   = datasets.ImageFolder(str(VAL_DIR),   transform=val_transforms)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    weights = EfficientNet_B0_Weights.DEFAULT
    model = efficientnet_b0(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(train_dataset.classes))
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda"))

    best_acc = 0.0

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for images, labels in tqdm(train_loader, desc="train"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda")):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += labels.size(0)

        print(f"Train Loss: {train_loss/len(train_loader):.4f}  Acc: {correct/total:.4f}")

        #val
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="val"):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, preds = outputs.max(1)
                correct += preds.eq(labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        print(f"Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"✅ Saved best model (val_acc={val_acc:.4f})")

        scheduler.step()

        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    print(f"\nTraining complete. Best val acc: {best_acc:.4f}")

    # Run prediction + TTA
    predict_script = ROOT / "predict_efficientnet.py"
    print(f"\nRunning {predict_script.name} ...")
    subprocess.run([sys.executable, str(predict_script)], check=True)
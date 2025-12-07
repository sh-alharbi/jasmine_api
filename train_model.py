# Training Script for Jasmine application


import os
import json
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from timm import create_model
import requests
from io import BytesIO
from sklearn.metrics import accuracy_score, f1_score

CSV_PATH = "fitzpatrick17k.csv"
MODEL_PATH = "/content/drive/MyDrive/jasmine_skin_model.pth"
CLASSES_PATH = "/content/drive/MyDrive/jasmine_classes.json"

IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4
NUM_WORKERS = 0


class FitzpatrickDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.data = pd.read_csv(csv_path)
        self.transform = transform

        label_col = 'label' if 'label' in self.data.columns else 'dx'
        self.labels = sorted(self.data[label_col].dropna().unique())
        self.label_to_idx = {l: i for i, l in enumerate(self.labels)}
        self.data['label_idx'] = self.data[label_col].map(self.label_to_idx)

        if 'url' in self.data.columns:
            self.image_col = 'url'
        elif 'image' in self.data.columns:
            self.image_col = 'image'
        else:
            raise ValueError("No image column found.")

        self.data = self.data.dropna(subset=[self.image_col])
        self.data = self.data[self.data[self.image_col].apply(
            lambda x: isinstance(x, str) and x.strip() != ""
        )]

        print(
            f"Dataset loaded: {len(self.data)} samples, {len(self.labels)} classes")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = str(row[self.image_col]).strip()

        try:
            if img_path.startswith("http"):
                response = requests.get(img_path, timeout=10)
                image = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                image = Image.open(img_path).convert("RGB")
        except Exception:
            return self.__getitem__((idx + 1) % len(self.data))

        if self.transform:
            image = self.transform(image)

        label = int(row["label_idx"])
        return image, label


def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        )
    ])

    dataset = FitzpatrickDataset(CSV_PATH, transform=transform)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )

    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )

    model = create_model(
        "tf_efficientnetv2_b3",
        pretrained=True,
        num_classes=len(dataset.labels)
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    best_acc = 0.0
    best_f1 = 0.0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"EPOCH {epoch+1} ")
        print(f"Loss: {avg_loss:.4f}")

        model.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, pred = torch.max(outputs.data, 1)

                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        acc = accuracy_score(all_labels, all_preds) * 100
        f1 = f1_score(all_labels, all_preds,
                      average="macro", zero_division=0) * 100

        print(f"Accuracy: {acc:.2f}%")
        print(f"F1 Score: {f1:.2f}%")

        if acc > best_acc:
            best_acc = acc
            best_f1 = f1
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Saved model  {MODEL_PATH}")

    with open(CLASSES_PATH, "w") as f:
        json.dump(dataset.labels, f)

    print("Training Completed!")
    print(f"Best Accuracy: {best_acc:.2f}%")
    print(f"Best F1 Score: {best_f1:.2f}%")


train_model()

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from timm import create_model
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from PIL import Image
import requests
from io import BytesIO

from train_model import CSV_PATH, MODEL_PATH, IMAGE_SIZE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class SimpleFitzpatrickDataset(Dataset):
    def __init__(self, df, transform=None):
        self.data = df.reset_index(drop=True)
        self.transform = transform

        self.image_col = "url" if "url" in self.data.columns else "image"

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

        label_idx = int(row["label_idx"])
        return image, label_idx


df = pd.read_csv(CSV_PATH)

label_col = "label" if "label" in df.columns else "dx"

labels = sorted(df[label_col].dropna().unique())
label_to_idx = {l: i for i, l in enumerate(labels)}
df["label_idx"] = df[label_col].map(label_to_idx)
df = df.dropna(subset=[label_col])

# نفس الـ split: 80% تدريب / 20% اختبار
_, test_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["label_idx"],
    random_state=42,
)

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225],
    ),
])

test_ds = SimpleFitzpatrickDataset(test_df, transform=test_transform)
test_loader = DataLoader(
    test_ds,
    batch_size=32,
    shuffle=False,
    num_workers=0,
)

print(f"Test samples: {len(test_ds)}")

num_classes = len(labels)


model = create_model(
    "tf_efficientnetv2_b3",
    pretrained=False,
    num_classes=num_classes,
).to(device)

state_dict = torch.load(MODEL_PATH, map_location=device)

if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
    state_dict = state_dict["model_state_dict"]

model.load_state_dict(state_dict, strict=False)
model.eval()

print(f"Loaded model weights from {MODEL_PATH}")
all_preds = []
all_labels = []

with torch.no_grad():
    for imgs, labels_batch in tqdm(test_loader, desc="Evaluating", ncols=80):
        imgs = imgs.to(device)
        labels_batch = labels_batch.to(device)

        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels_batch.cpu().numpy())


acc = accuracy_score(all_labels, all_preds) * 100
precision = precision_score(all_labels, all_preds,
                            average="macro", zero_division=0) * 100
recall = recall_score(all_labels, all_preds,
                      average="macro", zero_division=0) * 100
f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0) * 100

print("MODEL EVALUATION")
print(f"Accuracy : {acc:.2f}%")
print(f"Precision: {precision:.2f}%")
print(f"Recall   : {recall:.2f}%")
print(f"F1-score : {f1:.2f}%")

print(" Classification Report")
print(
    classification_report(
        all_labels,
        all_preds,
        target_names=list(labels),
        zero_division=0,
    )
)

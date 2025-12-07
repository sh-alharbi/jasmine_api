from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
import os
import json
import io
import requests
from typing import List, Dict

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from timm import create_model
from dotenv import load_dotenv

load_dotenv()
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


@app.get("/")
def root():
    return {"ok": True}


@app.get("/ping")
def ping():
    return {"pong": True}


CLASSES_PATH = "classes.json"
MODEL_PATH = "model.pth"
IMAGE_SIZE = 224
DEVICE = torch.device("cpu")

with open(CLASSES_PATH, "r") as f:
    CLASS_NAMES: List[str] = json.load(f)

model = create_model("tf_efficientnet_b0",
                     pretrained=False, num_classes=len(CLASS_NAMES))

state = torch.load(MODEL_PATH, map_location=DEVICE)
if isinstance(state, dict) and "state_dict" in state:
    state = state["state_dict"]
if isinstance(state, dict):
    state = {k.replace("module.", ""): v for k, v in state.items()}

model.load_state_dict(state, strict=False)
model.to(DEVICE).eval()


def transform_image(image_bytes: bytes) -> torch.Tensor:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    t = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])
    return t(img).unsqueeze(0).to(DEVICE)


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def call_llm_structured(prediction: str) -> str:
    if not OPENAI_API_KEY:
        return ""
    try:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        prompt = (
            f"The AI model predicted the skin condition as '{prediction}'.\n"
            "Please provide:\n"
            "1) A short, simple explanation of the condition.\n"
            "2) General prevention & lifestyle tips.\n"
            "3) 2–3 trusted medical sources (WHO, Mayo Clinic, NIH, WebMD).\n"
            "Keep it concise and non-diagnostic."
        )
        data = {
            "model": OPENAI_MODEL,
            "messages": [
                {"role": "system", "content": "You are a helpful dermatology assistant."},
                {"role": "user", "content": prompt},
            ],
            "max_tokens": 600,
            "temperature": 0.7,
        }
        resp = requests.post(url, headers=headers, json=data, timeout=60)
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"]
        return f"(LLM error {resp.status_code})"
    except Exception as e:
        return f"(LLM exception: {e})"


@app.post("/predict")
async def predict(file: UploadFile = File(...), topk: int = Query(1, ge=1, le=10)):
    image_bytes = await file.read()
    x = transform_image(image_bytes)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]

    top1_idx = int(torch.argmax(probs).item())
    top1 = {
        "label": CLASS_NAMES[top1_idx],
        "score": float(probs[top1_idx].item()),
    }

    k = min(topk, len(CLASS_NAMES))
    vals, idxs = torch.topk(probs, k=k)
    topk_list = [
        {"label": CLASS_NAMES[int(i.item())], "score": float(v.item())}
        for v, i in zip(vals, idxs)
    ]

    chat_explanation = call_llm_structured(top1["label"])

    return {
        "arch": "tf_efficientnet_b0",
        "num_classes": len(CLASS_NAMES),
        "top1": top1,
        "topk": topk_list,
        "chatgpt_explanation": chat_explanation,
    }

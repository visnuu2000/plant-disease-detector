import streamlit as st
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image

CLASS_NAMES = [
    "Apple - Apple scab", "Apple - Black rot",
    "Apple - Cedar apple rust", "Apple - Healthy",
    "Blueberry - Healthy", "Cherry - Powdery mildew",
    "Cherry - Healthy", "Corn - Cercospora leaf spot",
    "Corn - Common rust", "Corn - Northern Leaf Blight",
    "Corn - Healthy", "Grape - Black rot",
    "Grape - Esca", "Grape - Leaf blight",
    "Grape - Healthy", "Orange - Haunglongbing",
    "Peach - Bacterial spot", "Peach - Healthy",
    "Pepper - Bacterial spot", "Pepper - Healthy",
    "Potato - Early blight", "Potato - Late blight",
    "Potato - Healthy", "Raspberry - Healthy",
    "Soybean - Healthy", "Squash - Powdery mildew",
    "Strawberry - Leaf scorch", "Strawberry - Healthy",
    "Tomato - Bacterial spot", "Tomato - Early blight",
    "Tomato - Late blight", "Tomato - Leaf mold",
    "Tomato - Septoria leaf spot",
    "Tomato - Spider mites", "Tomato - Target spot",
    "Tomato - Yellow leaf curl virus",
    "Tomato - Mosaic virus", "Tomato - Healthy"
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

@st.cache_resource
def load_model():
    model = models.resnet50(weights="IMAGENET1K_V1")
    model.fc = torch.nn.Linear(
        model.fc.in_features, 38
    )
    model.eval()
    return model

def predict(image, model):
    if image.mode != "RGB":
        image = image.convert("RGB")
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.nn.functional.softmax(
            outputs[0], dim=0
        )
        top5 = torch.topk(probs, 5)
    results = []
    for i in range(5):
        idx = top5.indices[i].item()
        conf = round(top5.values[i].item() * 100, 1)
        results.append({
            "label": CLASS_NAMES[idx],
            "confidence": conf
        })
    return results
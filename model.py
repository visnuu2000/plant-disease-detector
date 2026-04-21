import streamlit as st
from transformers import AutoFeatureExtractor, AutoModelForImageClassification
from PIL import Image
import torch

@st.cache_resource
def load_model():
    extractor = AutoFeatureExtractor.from_pretrained(
        "linkanjarad/mobilenet-v2-1.0-224-plant-disease-identification"
    )
    model = AutoModelForImageClassification.from_pretrained(
        "linkanjarad/mobilenet-v2-1.0-224-plant-disease-identification"
    )
    model.eval()
    return extractor, model

def predict(image, extractor, model):
    if image.mode != "RGB":
        image = image.convert("RGB")
    inputs = extractor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(
            outputs.logits[0], dim=0
        )
        top5 = torch.topk(probs, 5)
    results = []
    for i in range(5):
        idx = top5.indices[i].item()
        label = model.config.id2label[idx]
        conf = round(top5.values[i].item() * 100, 1)
        results.append({
            "label": label,
            "confidence": conf
        })
    return results
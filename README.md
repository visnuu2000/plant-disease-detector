---
title: Plant Disease Detector
emoji: 🌿
colorFrom: green
colorTo: indigo
sdk: streamlit
sdk_version: 1.35.0
app_file: app.py
pinned: false
---

# Plant Disease Detector

Upload a photo of a plant leaf to detect diseases using
a ResNet-50 deep learning model trained on 38 disease classes.

## Tech stack
- Python, PyTorch, TorchVision
- ResNet-50 (CNN)
- Streamlit, Pillow

## How to run locally
pip install -r requirements.txt
streamlit run app.py

## Live demo
https://vishnuvardhanreddy12-plant-disease-detector.hf.space/
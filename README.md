# Plant Disease Detector

Upload a photo of a plant leaf. The app detects
if it is healthy or diseased using a ResNet model
trained on the PlantVillage dataset (54,306 images,
38 disease classes).

## Tech stack
- Python, PyTorch, TorchVision
- ResNet-50 (pre-trained, transfer learning)
- Streamlit, Pillow

## How to run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Live demo
[View on Hugging Face Spaces](PASTE_YOUR_LINK_HERE)
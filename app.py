import streamlit as st
from PIL import Image
from model import load_model, predict

st.set_page_config(page_title="Plant Disease Detector", layout="centered")
st.title("Plant Disease Detector")
st.write("Upload a photo of a plant leaf to detect diseases.")

with st.spinner("Loading model..."):
    model = load_model()

st.success("Model loaded!")

uploaded = st.file_uploader(
    "Upload a leaf image (JPG or PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded:
    image = Image.open(uploaded)
    st.image(image, caption="Uploaded image", width=300)

    if st.button("Detect disease"):
        with st.spinner("Analysing..."):
            results = predict(image, model)

        top = results[0]
        st.subheader("Results")

        if "healthy" in top["label"].lower():
            st.success("Plant appears healthy!")
        else:
            st.error("Disease detected!")

        st.metric("Prediction", top["label"])
        st.metric("Confidence", str(top["confidence"]) + "%")

        st.subheader("Top 5 predictions")
        for r in results:
            st.write(r["label"] + " — " + str(r["confidence"]) + "%")
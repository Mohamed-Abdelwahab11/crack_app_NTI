import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown

# ------------------------------------------------------------------
# Config
# ------------------------------------------------------------------
IMG_SIZE = (224, 224)
CLASS_NAMES = ["Negative", "Positive"]
MODEL_PATH = "crack_model.keras"

# Google Drive File ID للموديل (128MB)
GDRIVE_FILE_ID = "1DlAOA56mjR8Pnh3gsxJl1hYUpufvMJm3"

st.set_page_config(
    page_title="Surface Crack Detection",
    page_icon="🧱",
    layout="centered"
)


def download_model():
    """Download model from Google Drive if not present or is an LFS pointer."""
    if os.path.exists(MODEL_PATH):
        # Check if it's a real model or just a Git LFS pointer (< 1KB)
        if os.path.getsize(MODEL_PATH) < 1000:
            os.remove(MODEL_PATH)
        else:
            return  # Model already exists

    with st.spinner("Downloading model... please wait (~128 MB)"):
        url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
        gdown.download(url, MODEL_PATH, quiet=False)


@st.cache_resource
def load_model():
    download_model()
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    arr = np.array(image, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def main():
    st.title("🧱 Surface Crack Detection")
    st.write("Upload a concrete surface image and the model will detect whether it has a crack or not.")

    model = load_model()

    uploaded_file = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        with st.spinner("Analyzing..."):
            processed = preprocess_image(image)
            prediction = model.predict(processed)[0][0]

        predicted_class = CLASS_NAMES[1] if prediction >= 0.5 else CLASS_NAMES[0]
        confidence = prediction if prediction >= 0.5 else 1 - prediction

        st.subheader("Result")
        if predicted_class == "Positive":
            st.error(f"⚠️ Crack Detected — Confidence: {confidence:.2%}")
        else:
            st.success(f"✅ No Crack Detected — Confidence: {confidence:.2%}")

        with st.expander("Technical Details"):
            st.write(f"Raw sigmoid output: {prediction:.4f}")
            st.write(f"Threshold: 0.5")


if __name__ == "__main__":
    main()

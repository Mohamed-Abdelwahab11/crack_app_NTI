import streamlit as st
import onnxruntime as ort
import numpy as np
from PIL import Image
import os

# ------------------------------------------------------------------
# Config — input shape from model: (1, 227, 227, 3)
# ------------------------------------------------------------------
IMG_SIZE = (227, 227)
CLASS_NAMES = ["Negative", "Positive"]
MODEL_PATH = "crack_model.onnx"

st.set_page_config(
    page_title="Surface Crack Detection",
    page_icon="🧱",
    layout="centered"
)


@st.cache_resource
def load_model():
    session = ort.InferenceSession(MODEL_PATH)
    return session


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    arr = np.array(image, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr


def main():
    st.title("🧱 Surface Crack Detection")
    st.write("Upload a concrete surface image and the model will detect whether it has a crack or not.")

    session = load_model()

    uploaded_file = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        with st.spinner("Analyzing..."):
            processed = preprocess_image(image)
            input_name = session.get_inputs()[0].name
            output = session.run(None, {input_name: processed})
            prediction = float(output[0][0][0])

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

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ------------------------------------------------------------------
# Config - matches how the model was trained in the notebook
# ------------------------------------------------------------------
IMG_SIZE = (224, 224)
CLASS_NAMES = ["Negative", "Positive"]  # 0 = Negative (no crack), 1 = Positive (crack)
MODEL_PATH = "crack_model.keras"  # change if you saved with a different name

st.set_page_config(page_title="Surface Crack Detection", page_icon="🧱", layout="centered")


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    arr = np.array(image, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=0)  # batch dimension -> (1, 224, 224, 3)
    return arr


def main():
    st.title("🧱 Surface Crack Detection")
    st.write(
        "ارفع صورة لسطح خرساني (concrete surface) وهيقولك الموديل فيه كراك ولا لأ."
    )

    model = load_model()

    uploaded_file = st.file_uploader(
        "اختار صورة", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="الصورة اللي رفعتها", use_container_width=True)

        with st.spinner("جاري التحليل..."):
            processed = preprocess_image(image)
            prediction = model.predict(processed)[0][0]  # sigmoid output, single value

        predicted_class = CLASS_NAMES[1] if prediction >= 0.5 else CLASS_NAMES[0]
        confidence = prediction if prediction >= 0.5 else 1 - prediction

        st.subheader("النتيجة")
        if predicted_class == "Positive":
            st.error(f"⚠️ فيه كراك (Crack Detected) — Confidence: {confidence:.2%}")
        else:
            st.success(f"✅ مفيش كراك (No Crack) — Confidence: {confidence:.2%}")

        with st.expander("تفاصيل تقنية"):
            st.write(f"Raw sigmoid output: {prediction:.4f}")
            st.write(f"Threshold: 0.5")


if __name__ == "__main__":
    main()

import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# Load model
model = load_model("brain_tumor_model.h5")

# Class labels
classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

# Page Configuration
st.set_page_config(page_title="NeuroScan AI", page_icon="🧠", layout="centered")

# Sidebar Info
with st.sidebar:
    st.title("🧠 NeuroScan AI")
    st.markdown("**Smart Brain Tumor Classifier**")
    st.markdown("Upload an MRI image, and our AI will classify the tumor type with high confidence.")
    st.info("Supported Classes:\n- Glioma\n- Meningioma\n- Pituitary\n- No Tumor")
    st.markdown("---")
    st.caption("Built with ❤️ using Transfer Learning & Streamlit")

# Main Title
st.markdown(
    "<h1 style='text-align: center; color: #4B8BBE;'>Brain Tumor MRI Image Classifier</h1>",
    unsafe_allow_html=True
)

st.write("### 📤 Upload an MRI Image")

# Upload Image
uploaded_file = st.file_uploader("Drop or browse image here...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded MRI Image", use_column_width=True)

    # Preprocess
    img = image.resize((224, 224))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0]
    predicted_class = classes[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.markdown("---")
    st.subheader("🎯 Prediction Result")
    st.success(f"**Tumor Type:** {predicted_class}")
    st.info(f"**Model Confidence:** {confidence*100:.2f}%")

    # Show all scores
    with st.expander("📊 Show Confidence for All Classes"):
        for i, score in enumerate(prediction):
            st.write(f"🔹 {classes[i]}: {score*100:.2f}%")

# Footer
st.markdown("---")
st.caption("© 2025 NeuroScan AI | Medical Imaging Powered by Deep Learning")

import streamlit as st
import pandas as pd
import numpy as np
import cv2
import tempfile
import os
from PIL import Image
from ultralytics import YOLO
import uuid

# Load YOLOv8 models only once and cache them
@st.cache_resource
def load_models():
    model_map = {
        "Nano (Eco Mode)": YOLO("models/Nano/weights/best.pt"),
        "Small": YOLO("models/Small/weights/best.pt"),
        "Medium": YOLO("models/Medium/weights/best.pt")
    }
    return model_map

model_map = load_models()

# App Config
st.set_page_config(page_title="🛰 Multi-Model Multi-Image Detection Dashboard", layout="wide")
st.title("🛰 Upload Multiple Images for Detection")

# Sidebar
selected_model = st.sidebar.selectbox("Choose YOLOv8 Model", list(model_map.keys()))

# Upload multiple images
uploaded_files = st.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    model = model_map[selected_model]

    st.subheader("Detection Results")
    for uploaded_file in uploaded_files:
        # Read and convert uploaded file to PIL image
        image = Image.open(uploaded_file).convert("RGB")

        # Save to a temp file for YOLO inference
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            image.save(tmp_file.name)
            temp_image_path = tmp_file.name

        # Perform detection
        results = model(temp_image_path)

        # Generate a unique filename
        output_filename = f"{uuid.uuid4().hex}.jpg"
        output_path = os.path.join(output_dir, output_filename)

        # Save result image
        results[0].save(filename=output_path)

        # Show original + result
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption=f"Original - {uploaded_file.name}", use_container_width=True)
        with col2:
            st.image(output_path, caption="Detection Result", use_container_width=True)

        # Clean up temp file
        os.remove(temp_image_path)

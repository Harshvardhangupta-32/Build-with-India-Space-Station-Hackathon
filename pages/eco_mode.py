import streamlit as st
from PIL import Image
import time
import psutil
import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Load all YOLOv8 models
@st.cache_resource
def load_models():
    models = {
        "Nano (Eco Mode)": YOLO("models/Nano/weights/best.pt"),
        "Small": YOLO("models/Small/weights/best.pt"),
        "Medium": YOLO("models/Medium/weights/best.pt")
    }
    for model in models.values():
        model.fuse()
    return models

models = load_models()

st.set_page_config(page_title="YOLO Model Comparison Benchmark", layout="centered")
st.title("📊 YOLOv8 Model Resource Usage Comparison")

# Upload an image
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    # Read and display the uploaded image
    file_bytes = uploaded_file.read()
    np_img = np.frombuffer(file_bytes, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    st.image(image_rgb, caption="Original Image", use_container_width=True)

    # Track metrics
    metrics = []

    for name, model in models.items():
        st.subheader(f"🔍 Results using {name}")

        # Resource usage before inference
        process = psutil.Process(os.getpid())
        cpu_before = psutil.cpu_percent(interval=1)
        mem_before = process.memory_info().rss / (1024 * 1024)

        # Inference timing
        start_time = time.time()
        results = model.predict(frame, imgsz=416, conf=0.25, verbose=False)
        end_time = time.time()

        # Resource usage after inference
        cpu_after = psutil.cpu_percent(interval=1)
        mem_after = process.memory_info().rss / (1024 * 1024)

        # Display results
        res_img = results[0].plot()
        st.image(res_img, caption=f"Detected Objects - {name}", use_container_width=True)

        inference_time = end_time - start_time
        avg_cpu = (cpu_before + cpu_after) / 2
        ram_usage = mem_after - mem_before

        metrics.append((name, inference_time, avg_cpu, ram_usage))

    # Create DataFrame
    df = pd.DataFrame(metrics, columns=["Model", "Inference Time (s)", "Avg CPU Usage (%)", "RAM Usage Increase (MB)"])

    # Comparison Table
    st.subheader("📈 Comparison Summary")
    st.dataframe(df, use_container_width=True)

    # Bar Charts
    st.subheader("📊 Visual Comparison")

    # Inference Time
    fig1, ax1 = plt.subplots()
    ax1.bar(df["Model"], df["Inference Time (s)"], color='skyblue')
    ax1.set_title("Inference Time")
    ax1.set_ylabel("Seconds")
    st.pyplot(fig1) 

    # CPU Usage
    fig2, ax2 = plt.subplots()
    ax2.bar(df["Model"], df["Avg CPU Usage (%)"], color='orange')
    ax2.set_title("CPU Usage")
    ax2.set_ylabel("Percentage (%)")
    st.pyplot(fig2)

    # RAM Usage
    fig3, ax3 = plt.subplots()
    ax3.bar(df["Model"], df["RAM Usage Increase (MB)"], color='green')
    ax3.set_title("RAM Usage Increase")
    ax3.set_ylabel("MB")
    st.pyplot(fig3)

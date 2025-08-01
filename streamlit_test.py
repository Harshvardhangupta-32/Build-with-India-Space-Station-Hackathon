import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import time
<<<<<<< HEAD
from ultralytics import YOLO

# --------- Load YOLOv8 Model ---------
@st.cache_resource
def load_model():
    return YOLO("runs/detect/train2/weights/best.pt")  # Update path as needed

model = load_model()

# --------- UI Config ---------
st.set_page_config(page_title="🚀 Multi-Cam Detection Dashboard", layout="wide")
st.sidebar.title("🎛️ Control Panel")
st.sidebar.markdown("Configure camera inputs and detection threshold.")
confidence_threshold = st.sidebar.slider("Detection Confidence Threshold", 0.0, 1.0, 0.5, 0.05)

upload_cam1 = st.sidebar.file_uploader("Upload Image from Camera 1", type=["png", "jpg", "jpeg"], key="cam1")
upload_cam2 = st.sidebar.file_uploader("Upload Image from Camera 2", type=["png", "jpg", "jpeg"], key="cam2")

st.title("🛰️ Multi-Camera Object Detection App")
st.caption("Powered by Falcon synthetic dataset + YOLOv8")

col1, col2 = st.columns(2)

# --------- Show Uploaded Images ---------
=======
import random

st.set_page_config(page_title="🚀 Multi-Cam Detection Dashboard", layout="wide")

# ----- Sidebar -----
st.sidebar.title("🎛️ Control Panel")
st.sidebar.markdown("Configure camera inputs and detection threshold.")

confidence_threshold = st.sidebar.slider("Detection Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
upload_cam1 = st.sidebar.file_uploader("Upload Image from Camera 1", type=["png", "jpg", "jpeg"], key="cam1")
upload_cam2 = st.sidebar.file_uploader("Upload Image from Camera 2", type=["png", "jpg", "jpeg"], key="cam2")

# ----- Header -----
st.title("🛰️ Multi-Camera Object Detection App")
st.caption("Powered by Falcon synthetic dataset + YOLOv8 ensemble logic")

# ----- Image Display -----
col1, col2 = st.columns(2)

>>>>>>> c8e1eb86b5ebdd5c5687d369c44ff181e4786cfe
with col1:
    st.subheader("📷 Camera 1 View")
    if upload_cam1:
        img1 = Image.open(upload_cam1)
        st.image(img1, use_column_width=True)
    else:
        st.info("Upload an image for Camera 1.")

with col2:
    st.subheader("📷 Camera 2 View")
    if upload_cam2:
        img2 = Image.open(upload_cam2)
        st.image(img2, use_column_width=True)
    else:
        st.info("Upload an image for Camera 2.")

<<<<<<< HEAD
# --------- Perform Real Detection ---------
if upload_cam1 and upload_cam2:
    st.markdown("---")
    st.header("🧠 Detection Fusion & Analysis")
    with st.spinner("Running YOLOv8 detection on both views..."):
        img1_np = np.array(img1.convert("RGB"))
        img2_np = np.array(img2.convert("RGB"))

        results1 = model.predict(img1_np, conf=confidence_threshold)
        results2 = model.predict(img2_np, conf=confidence_threshold)

        boxes1 = results1[0].boxes
        boxes2 = results2[0].boxes

        detections_cam1 = []
        detections_cam2 = []

        # Parse results from camera 1
        if boxes1 is not None:
            for i in range(len(boxes1.cls)):
                label = model.names[int(boxes1.cls[i])]
                conf = float(boxes1.conf[i])
                if conf >= confidence_threshold:
                    detections_cam1.append({
                        "label": label,
                        "confidence": round(conf, 2),
                        "bbox": boxes1.xyxy[i].tolist()
                    })

        # Parse results from camera 2
        if boxes2 is not None:
            for i in range(len(boxes2.cls)):
                label = model.names[int(boxes2.cls[i])]
                conf = float(boxes2.conf[i])
                if conf >= confidence_threshold:
                    detections_cam2.append({
                        "label": label,
                        "confidence": round(conf, 2),
                        "bbox": boxes2.xyxy[i].tolist()
                    })

    # --------- Raw Detections ---------
    st.subheader("🔍 Raw Detections")
    st.json({"Camera 1": detections_cam1, "Camera 2": detections_cam2})

    # --------- Fusion Logic ---------
    st.subheader("🗳️ Fusion Outcome")
    fused_results = []
    labels_cam1 = {d['label'] for d in detections_cam1}
    labels_cam2 = {d['label'] for d in detections_cam2}
=======
# ----- Simulate Detection and Merge Logic -----
if upload_cam1 and upload_cam2:
    st.markdown("---")
    st.header("🧠 Detection Fusion & Analysis")
    
    with st.spinner("Running YOLOv8 object detection on both views..."):
        time.sleep(2)  # Simulate processing delay

    # Simulated mock results (in real app, replace with actual YOLO inference)
    detections_cam1 = [{"label": "Toolbox", "confidence": 0.92, "bbox": [50, 80, 200, 250]}]
    detections_cam2 = [{"label": "Toolbox", "confidence": 0.88, "bbox": [48, 85, 198, 255]},
                       {"label": "Fire Extinguisher", "confidence": 0.78, "bbox": [250, 100, 350, 300]}]

    st.subheader("🔍 Raw Detections")
    st.json({"Camera 1": detections_cam1, "Camera 2": detections_cam2})

    # Apply simple voting mechanism
    st.subheader("🗳️ Fusion Outcome")
    fused_results = []
    labels_cam1 = {d['label'] for d in detections_cam1 if d['confidence'] >= confidence_threshold}
    labels_cam2 = {d['label'] for d in detections_cam2 if d['confidence'] >= confidence_threshold}
>>>>>>> c8e1eb86b5ebdd5c5687d369c44ff181e4786cfe
    final_labels = labels_cam1.union(labels_cam2)

    for label in final_labels:
        c1 = next((d for d in detections_cam1 if d['label'] == label), None)
        c2 = next((d for d in detections_cam2 if d['label'] == label), None)
        confs = [d['confidence'] for d in [c1, c2] if d]
        avg_conf = sum(confs) / len(confs)
        fused_results.append({"label": label, "avg_confidence": round(avg_conf, 2)})

    st.success("✅ Objects detected with multi-camera fusion:")
    st.json(fused_results)

<<<<<<< HEAD
    # --------- Benefits Section ---------
=======
    # Benefits Panel
>>>>>>> c8e1eb86b5ebdd5c5687d369c44ff181e4786cfe
    st.markdown("""
    ### 🚀 Why Multi-Camera?
    - Detect occluded objects more reliably
    - Improve accuracy via viewpoint consensus
    - Mimics real-world space station camera layouts
    """)
else:
    st.warning("Please upload images from both cameras to proceed with detection.")

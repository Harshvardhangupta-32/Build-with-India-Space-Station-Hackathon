import streamlit as st
import cv2
import tempfile
import time
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Load YOLOv8 model
@st.cache_resource
def load_model():
    return YOLO("runs/detect/train2/weights/best.pt")

model = load_model()

# App Config
st.set_page_config(page_title="🛰️ Space Object Detection Dashboard", layout="wide")
st.title("🛰️ Space Detection Suite with Multi-Camera Fusion")

# Sidebar Input Mode
mode = st.sidebar.radio("Choose Input Mode", ["Webcam", "Upload Video", "Single Image", "Multi-Camera Fusion"])
confidence = st.sidebar.slider("Confidence Threshold", 0.25, 1.0, 0.5, 0.05)

# Utility: Display Detection Stats
def display_stats(results):
    boxes = results[0].boxes
    if boxes is not None:
        st.write(f"✅ Total Detections: {len(boxes)}")
        counts = {}
        for c in boxes.cls:
            label = model.names[int(c)]
            counts[label] = counts.get(label, 0) + 1
        st.write("🔍 Class Counts:")
        st.json(counts)

# Utility: Run detection on frame/image
def detect_and_annotate(image_np):
    results = model.predict(image_np, conf=confidence)
    annotated = results[0].plot()
    return annotated, results

# Webcam or video mode
def run_video_detection(video_source):
    stframe = st.empty()
    cap = cv2.VideoCapture(video_source)
    prev_time = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        annotated_frame, results = detect_and_annotate(frame)
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time + 1e-4)
        prev_time = curr_time
        stframe.image(annotated_frame, channels="BGR", use_column_width=True)
        st.caption(f"FPS: {fps:.2f}")
        display_stats(results)

    cap.release()

# Image mode
def run_image_detection(uploaded_image):
    img = Image.open(uploaded_image).convert("RGB")
    img_np = np.array(img)
    annotated, results = detect_and_annotate(img_np)
    st.image(img, caption="Original Image", use_column_width=True)
    st.image(annotated, caption="Detected Image", use_column_width=True)
    display_stats(results)

# Multi-Cam Fusion
def run_multi_camera_fusion(img1, img2):
    img1_np = np.array(img1.convert("RGB"))
    img2_np = np.array(img2.convert("RGB"))

    results1 = model.predict(img1_np, conf=confidence)
    results2 = model.predict(img2_np, conf=confidence)

    boxes1 = results1[0].boxes
    boxes2 = results2[0].boxes

    detections_cam1, detections_cam2 = [], []

    for i in range(len(boxes1.cls)):
        conf = float(boxes1.conf[i])
        if conf >= confidence:
            detections_cam1.append({
                "label": model.names[int(boxes1.cls[i])],
                "confidence": round(conf, 2),
                "bbox": boxes1.xyxy[i].tolist()
            })

    for i in range(len(boxes2.cls)):
        conf = float(boxes2.conf[i])
        if conf >= confidence:
            detections_cam2.append({
                "label": model.names[int(boxes2.cls[i])],
                "confidence": round(conf, 2),
                "bbox": boxes2.xyxy[i].tolist()
            })

    st.subheader("🔍 Raw Detections")
    st.json({"Camera 1": detections_cam1, "Camera 2": detections_cam2})

    # Fusion
    st.subheader("🗳️ Fusion Outcome")
    labels_cam1 = {d['label'] for d in detections_cam1}
    labels_cam2 = {d['label'] for d in detections_cam2}
    final_labels = labels_cam1.union(labels_cam2)
    fused_results = []

    for label in final_labels:
        c1 = next((d for d in detections_cam1 if d['label'] == label), None)
        c2 = next((d for d in detections_cam2 if d['label'] == label), None)
        confs = [d['confidence'] for d in [c1, c2] if d]
        avg_conf = sum(confs) / len(confs)
        fused_results.append({"label": label, "avg_confidence": round(avg_conf, 2)})

    st.success("✅ Fused Detections:")
    st.json(fused_results)

    st.markdown("""
    ### 🤖 Why Fusion?
    - Handles occlusion from one camera
    - Adds redundancy for critical detection
    - Mimics surveillance systems in ISS / robotics
    """)

# ========== Execution Logic ==========

if mode == "Webcam":
    st.warning("Ensure webcam permission is granted in browser.")
    run_video_detection(0)

elif mode == "Upload Video":
    video_file = st.file_uploader("🎥 Upload a video", type=["mp4", "avi", "mov"])
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        run_video_detection(tfile.name)

elif mode == "Single Image":
    image_file = st.file_uploader("🖼️ Upload an image", type=["jpg", "jpeg", "png"])
    if image_file:
        run_image_detection(image_file)

elif mode == "Multi-Camera Fusion":
    cam1 = st.file_uploader("Upload Camera 1 Image", type=["jpg", "jpeg", "png"], key="cam1")
    cam2 = st.file_uploader("Upload Camera 2 Image", type=["jpg", "jpeg", "png"], key="cam2")

    col1, col2 = st.columns(2)
    if cam1:
        with col1:
            st.image(cam1, caption="Camera 1 View", use_column_width=True)
    if cam2:
        with col2:
            st.image(cam2, caption="Camera 2 View", use_column_width=True)

    if cam1 and cam2:
        st.markdown("---")
        st.header("🔬 Detection Fusion Results")
        run_multi_camera_fusion(Image.open(cam1), Image.open(cam2))
    else:
        st.info("Upload images from both cameras to enable fusion.")

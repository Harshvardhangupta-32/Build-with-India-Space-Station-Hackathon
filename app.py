import streamlit as st
import cv2
import tempfile
import time
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Load model
@st.cache_resource
def load_model():
    return YOLO("runs/detect/train2/weights/best.pt")  # Update with your model path

model = load_model()

# App config
st.set_page_config(page_title="Space Detection Dashboard", layout="wide")
st.title("🛰️ Real-Time Space Object Detection")

# Sidebar controls
mode = st.sidebar.radio("Choose Input Mode", ["Webcam", "Upload Video", "Upload Image"])
confidence = st.sidebar.slider("Confidence Threshold", 0.25, 1.0, 0.5, 0.05)

# Stats printer
def display_stats(results):
    if not results:
        return

    boxes = results[0].boxes
    if boxes is not None:
        st.write(f"✅ Total Detections: {len(boxes)}")
        counts = {}
        for c in boxes.cls:
            label = model.names[int(c)]
            counts[label] = counts.get(label, 0) + 1
        st.write("🔍 Class Counts:")
        st.json(counts)

# Webcam or video detection
def run_video_detection(video_source):
    stframe = st.empty()
    cap = cv2.VideoCapture(video_source)
    prev_time = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=confidence)
        annotated_frame = results[0].plot()

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time + 1e-4)
        prev_time = curr_time

        stframe.image(annotated_frame, channels="BGR", use_column_width=True)
        st.caption(f"FPS: {fps:.2f}")
        display_stats(results)

    cap.release()

# Image detection
def run_image_detection(image):
    img_array = np.array(image.convert("RGB"))
    results = model.predict(img_array, conf=confidence)
    annotated_img = results[0].plot()

    st.image(annotated_img, caption="Detected Image", use_column_width=True)
    display_stats(results)

# Input Modes
if mode == "Webcam":
    st.warning("Ensure your webcam is enabled.")
    run_video_detection(0)

elif mode == "Upload Video":
    uploaded_file = st.file_uploader("🎥 Upload a video", type=["mp4", "mov", "avi"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        run_video_detection(tfile.name)

elif mode == "Upload Image":
    uploaded_image = st.file_uploader("🖼️ Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        img = Image.open(uploaded_image)
        st.image(img, caption="Original Image", use_column_width=True)
        run_image_detection(img)

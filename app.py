import streamlit as st
import pandas as pd
import numpy as np
import cv2
import tempfile
import time
from PIL import Image
from ultralytics import YOLO

# Load YOLOv8 model
@st.cache_resource
def load_model():
    return YOLO("runs/detect/train2/weights/best.pt")  # Update path if needed

model = load_model()

# ----- App Config -----
st.set_page_config(page_title="🛰️ Multi-Cam Detection Dashboard", layout="wide")
st.title("🛰️ Real-Time & Multi-Camera Object Detection")
st.caption("Powered by Falcon synthetic dataset + YOLOv8")

# ----- Sidebar -----
st.sidebar.title("🎛️ Control Panel")
mode = st.sidebar.radio("Choose Input Mode", ["Multi-Camera", "Webcam", "Upload Video", "Upload Image"])
confidence = st.sidebar.slider("Confidence Threshold", 0.25, 1.0, 0.5, 0.05)

# ----- Utility: Stats Printer -----
def display_stats(results):
    if not results:
        return
    boxes = results[0].boxes
    if boxes is not None and len(boxes.cls) > 0:
        st.write(f"✅ Total Detections: {len(boxes.cls)}")
        counts = {}
        for c in boxes.cls:
            label = model.names[int(c)]
            counts[label] = counts.get(label, 0) + 1
        st.write("🔍 Class Counts:")
        st.table(counts)

# ----- Utility: Detection -----
def run_image_detection(image):
    img_array = np.array(image.convert("RGB"))
    results = model.predict(img_array, conf=confidence)
    annotated_img = results[0].plot()
    st.image(annotated_img, caption="Detected Image")  # removed use_container_width=True
    display_stats(results)
    return results

def run_video_detection(video_source):
    stframe = st.empty()
    cap = cv2.VideoCapture(video_source)
    prev_time = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model.predict(frame, conf=confidence)
        annotated_frame = results[0].plot()

        curr_time = time.time()
        fps = 1 / max(curr_time - prev_time, 1e-4)
        prev_time = curr_time

        stframe.image(annotated_frame, channels="BGR", use_container_width=True)
        st.caption(f"FPS: {fps:.2f}")
        display_stats(results)

    cap.release()

# ----- Multi-Camera Mode -----
if mode == "Multi-Camera":
    st.subheader("📷 Upload from Multiple Cameras")
    upload_cam1 = st.sidebar.file_uploader("Upload Image from Camera 1", type=["png", "jpg", "jpeg"], key="cam1")
    upload_cam2 = st.sidebar.file_uploader("Upload Image from Camera 2", type=["png", "jpg", "jpeg"], key="cam2")

    col1, col2 = st.columns(2)

    if upload_cam1:
        with col1:
            st.subheader("Camera 1 View")
            img1 = Image.open(upload_cam1)
            st.image(img1)  # removed use_container_width=True
    else:
        col1.info("Upload an image for Camera 1.")

    if upload_cam2:
        with col2:
            st.subheader("Camera 2 View")
            img2 = Image.open(upload_cam2)
            st.image(img2)  # removed use_container_width=True
    else:
        col2.info("Upload an image for Camera 2.")

    if upload_cam1 and upload_cam2:
        st.markdown("---")
        st.header("🧠 Detection Fusion & Analysis")
        with st.spinner("Running YOLOv8 object detection on both views..."):
            results_cam1 = run_image_detection(img1)
            results_cam2 = run_image_detection(img2)

        def extract_detections(results):
            boxes = results[0].boxes
            data = []
            if boxes is not None:
                for i in range(len(boxes.cls)):
                    conf = float(boxes.conf[i])
                    cls = int(boxes.cls[i])
                    if conf >= confidence:
                        label = model.names[cls]
                        data.append({"label": label, "confidence": round(conf, 2)})
            return data

        det1 = extract_detections(results_cam1)
        det2 = extract_detections(results_cam2)

        st.subheader("🔍 Raw Detections")

        # Convert detections to DataFrames for tabular display
        df_cam1 = pd.DataFrame(det1)
        df_cam2 = pd.DataFrame(det2)

        st.write("Camera 1 Detections")
        st.dataframe(df_cam1)

        st.write("Camera 2 Detections")
        st.dataframe(df_cam2)

        # Fusion logic
        st.subheader("🗳️ Fusion Outcome")
        labels_cam1 = {d['label'] for d in det1}
        labels_cam2 = {d['label'] for d in det2}
        final_labels = labels_cam1.union(labels_cam2)

        fused_results = []
        for label in final_labels:
            confs = [d['confidence'] for d in det1 + det2 if d['label'] == label]
            avg_conf = sum(confs) / len(confs)
            fused_results.append({"label": label, "avg_confidence": round(avg_conf, 2)})

        st.success("✅ Objects detected with multi-camera fusion:")

        for i, obj in enumerate(fused_results):
            label = st.text_input(f"Label #{i+1}", obj['label'])
            avg_conf = st.number_input(
                f"Avg Confidence #{i+1}",
                min_value=0.0,
                max_value=1.0,
                value=obj['avg_confidence'],
                step=0.01,
                format="%.2f"
            )

# ----- Webcam Mode -----
elif mode == "Webcam":
    st.warning("Ensure your webcam is enabled and accessible.")
    run_video_detection(0)

# ----- Video Upload Mode -----
elif mode == "Upload Video":
    uploaded_file = st.file_uploader("🎥 Upload a video", type=["mp4", "mov", "avi"])
    if uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        run_video_detection(tfile.name)

# ----- Image Upload Mode -----
elif mode == "Upload Image":
    uploaded_images = st.file_uploader("🖼️ Upload image(s)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if uploaded_images:
        for uploaded_image in uploaded_images:
            img = Image.open(uploaded_image)
            st.image(img, caption=f"Original Image: {uploaded_image.name}")  # removed use_container_width=True
            run_image_detection(img)


import streamlit as st
import tempfile
import cv2
from utils import load_yolo_model, load_tracker, detect_and_track

st.set_page_config(page_title="YOLOv8 + DeepSORT", layout="wide")
st.title("🎯 YOLOv8 + DeepSORT Object Tracker")

# Upload video
uploaded_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])

if uploaded_file:
    st.video(uploaded_file)

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty()

    model = load_yolo_model()
    tracker = load_tracker()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = detect_and_track(frame, model, tracker)
        stframe.image(frame, channels="BGR", use_container_width=True)
    
    cap.release()

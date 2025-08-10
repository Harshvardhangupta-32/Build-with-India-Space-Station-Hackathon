import streamlit as st
import cv2
import tempfile
import time
from PIL import Image
from ultralytics import YOLO

# Load YOLOv8 models
@st.cache_resource
def load_models():
    return {
        "Nano (Eco Mode)": YOLO("models/Nano/weights/best.pt"),
        "Small": YOLO("models/Small/weights/best.pt"),
        "Medium": YOLO("models/Medium/weights/best.pt")
    }

models = load_models()

# App Config
st.set_page_config(page_title="🛰 Real-Time Detection", layout="wide")
st.title("📸 Real-Time Detection with Multiple Models")

# Sidebar for model and camera selection
model_name = st.sidebar.selectbox("Select YOLOv8 Model", list(models.keys()))
camera_index = st.sidebar.number_input("Select Camera Index", min_value=0, max_value=5, value=0, step=1)

model = models[model_name]

# Streamlit widgets
start_detection = st.button("Start Detection")
stop_detection = st.button("Stop Detection")
frame_placeholder = st.empty()

# Detection loop
if start_detection:
    cap = cv2.VideoCapture(camera_index)
    st.write(f"Using model: {model_name} | Camera Index: {camera_index}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.warning("Camera not found or cannot read frame.")
            break

        # Convert frame to RGB and save to temp file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        Image.fromarray(rgb_frame).save(temp_file.name)

        # Perform detection
        results = model(temp_file.name)
        annotated_frame = results[0].plot()

        # Show result in Streamlit
        frame_placeholder.image(annotated_frame, channels="BGR")

        # Break loop if stop button is pressed
        if stop_detection:
            break

        time.sleep(0.05)

    cap.release()

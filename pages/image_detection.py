import streamlit as st
from PIL import Image
from ultralytics import YOLO
import os
import time

st.title("📷 Image Detection with YOLOv8")

model_options = {
    "YOLOv8n (Nano)": r"D:\\NEW LEARNING\\HackwithIndia - 6 Aug\\HackByte_Dataset\\runs\\detect\\Nano\\weights\\best.pt",
    "YOLOv8s (Small)": r"D:\\NEW LEARNING\\HackwithIndia - 6 Aug\\HackByte_Dataset\\runs\\detect\\train2\\weights\\best.pt",
    "YOLOv8m (Medium)": r"D:\\NEW LEARNING\\HackwithIndia - 6 Aug\\HackByte_Dataset\\runs\\detect\\train4\\weights\\best.pt"
}

model_name = st.selectbox("Choose a YOLOv8 model:", list(model_options.keys()))
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("🚀 Run Detection"):
        with st.spinner("Running detection..."):
            input_path = f"input_{int(time.time())}.jpg"
            image.save(input_path)

            model = YOLO(model_options[model_name])
            results = model.predict(source=input_path, save=True)

            result_dir = results[0].save_dir
            for file in os.listdir(result_dir):
                if file.endswith(".jpg") or file.endswith(".png"):
                    st.image(os.path.join(result_dir, file), caption=f"Prediction with {model_name}", use_column_width=True)
                    break

            os.remove(input_path)

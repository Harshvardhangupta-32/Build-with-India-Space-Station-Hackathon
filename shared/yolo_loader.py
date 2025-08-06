from ultralytics import YOLO

def load_yolo_model(model_name):
    model_paths = {
        "YOLOv8n (Nano)": r"D:\NEW LEARNING\HackwithIndia - 6 Aug\HackByte_Dataset\runs\detect\Nano\weights\best.pt",
        "YOLOv8s (Small)": r"D:\NEW LEARNING\HackwithIndia - 6 Aug\HackByte_Dataset\runs\detect\train2\weights\best.pt",
        "YOLOv8m (Medium)": r"D:\NEW LEARNING\HackwithIndia - 6 Aug\HackByte_Dataset\runs\detect\train4\weights\best.pt"
    }

    model_path = model_paths[model_name]
    model = YOLO(model_path)
    return model

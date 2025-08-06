from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import cv2
import numpy as np

# Load YOLOv8 model
def load_yolo_model():
    return YOLO("models/Nano/weights/best.pt")  # Update to your path

# Load DeepSort Tracker
def load_tracker():
    return DeepSort(max_age=30)

# Process a single frame with YOLO + DeepSORT
def detect_and_track(frame, model, tracker):
    results = model(frame, verbose=False)[0]

    detections = []
    for r in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = r
        detections.append(([x1, y1, x2 - x1, y2 - y1], score, model.names[int(class_id)]))

    tracks = tracker.update_tracks(detections, frame=frame)

    for track in tracks:
        if not track.is_confirmed():
            continue
        ltrb = track.to_ltrb()
        track_id = track.track_id
        class_name = track.get_det_class()

        x1, y1, x2, y2 = map(int, ltrb)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (50, 255, 50), 2)
        cv2.putText(frame, f"{class_name} ID: {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 255, 50), 2)

    return frame

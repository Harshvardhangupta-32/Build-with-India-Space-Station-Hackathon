import argparse
from ultralytics import YOLO
import os

# Default hyperparameters
EPOCHS = 25
MOSAIC = 1.0
OPTIMIZER = 'AdamW'
MOMENTUM = 0.937
LR0 = 0.001
LRF = 0.01
SINGLE_CLS = False
IMG_SIZE = 640
BATCH = 10

# Data augmentation parameters
AUGMENTATION = {
    "degrees": 10,          # image rotation (± deg)
    "translate": 0.1,       # image translation (%)
    "scale": 0.5,           # image scale (± %)
    "shear": 0.2,           # image shear (± deg)
    "perspective": 0.001,   # perspective warping
    "flipud": 0.1,          # vertical flip
    "fliplr": 0.5,          # horizontal flip
    "hsv_h": 0.015,         # hue augmentation
    "hsv_s": 0.7,           # saturation augmentation
    "hsv_v": 0.4,           # brightness augmentation
    "mixup": 0.2            # mixup probability
}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=EPOCHS)
    parser.add_argument('--mosaic', type=float, default=MOSAIC)
    parser.add_argument('--optimizer', type=str, default=OPTIMIZER)
    parser.add_argument('--momentum', type=float, default=MOMENTUM)
    parser.add_argument('--lr0', type=float, default=LR0)
    parser.add_argument('--lrf', type=float, default=LRF)
    parser.add_argument('--single_cls', type=bool, default=SINGLE_CLS)
    parser.add_argument('--imgsz', type=int, default=IMG_SIZE)
    parser.add_argument('--batch', type=int, default=BATCH)
    args = parser.parse_args()

    # Ensure the script runs from its own directory
    this_dir = os.path.dirname(__file__)
    os.chdir(this_dir)

    # Load model
    model = YOLO(os.path.join(this_dir, "yolov8l.pt"))

    # Train
    results = model.train(
        data=os.path.join(this_dir, "yolo_params.yaml"),
        epochs=args.epochs,
        device=0,
        single_cls=args.single_cls,
        mosaic=args.mosaic,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        imgsz=args.imgsz,
        batch=args.batch,
        project="trained_model_outputs",
        name="multi_object_improved",
        verbose=True,
        patience=15,
        **AUGMENTATION  # Pass all augmentation values here
    )

    print("✅ Training completed. Results:", results)

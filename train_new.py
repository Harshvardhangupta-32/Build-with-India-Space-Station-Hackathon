import argparse
from ultralytics import YOLO
import os
import yaml
import shutil
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

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

# Data augmentation parameters (will be applied to all classes except class 2)
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

# Reduced augmentation for class 2 (minimal augmentation to preserve class characteristics)
CLASS_2_AUGMENTATION = {
    "degrees": 0,           # no rotation
    "translate": 0.02,      # minimal translation
    "scale": 0.1,           # minimal scale change
    "shear": 0,             # no shear
    "perspective": 0,       # no perspective warping
    "flipud": 0,            # no vertical flip
    "fliplr": 0.1,          # minimal horizontal flip
    "hsv_h": 0.005,         # minimal hue change
    "hsv_s": 0.1,           # minimal saturation change
    "hsv_v": 0.1,           # minimal brightness change
    "mixup": 0              # no mixup
}

def has_class_2(label_file):
    """Check if a label file contains class 2 annotations (OxygenTank)"""
    if not label_file.exists():
        return False
    
    try:
        with open(label_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if line and len(line.split()) >= 5:  # Valid YOLO format: class x y w h
                    class_id = int(line.split()[0])
                    if class_id == 2:  # OxygenTank
                        return True
    except Exception as e:
        print(f"⚠️  Error reading label file {label_file}: {e}")
    
    return False

def inspect_dataset_structure(this_dir):
    """Inspect and print the actual dataset structure"""
    print("🔍 Inspecting dataset structure...")
    
    # Look for common dataset patterns
    data_dir = os.path.join(this_dir, 'data')
    if os.path.exists(data_dir):
        print(f"📁 Found data directory: {data_dir}")
        for root, dirs, files in os.walk(data_dir):
            level = root.replace(data_dir, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = ' ' * 2 * (level + 1)
            for file in files[:5]:  # Show first 5 files
                print(f"{subindent}{file}")
            if len(files) > 5:
                print(f"{subindent}... and {len(files) - 5} more files")
    
    # Check yolo_params.yaml
    yaml_path = os.path.join(this_dir, "yolo_params.yaml")
    if os.path.exists(yaml_path):
        print(f"📋 Found yolo_params.yaml")
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"   Config: {config}")
    else:
        print("❌ yolo_params.yaml not found")

def create_class_specific_datasets(data_yaml_path, output_dir):
    """Split dataset into two parts: with class 2 and without class 2"""
    
    print("🔄 Creating class-specific datasets...")
    
    # Load original data configuration
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    print(f"📋 Data config loaded: {data_config}")
    
    # Create output directories
    output_dir = Path(output_dir)
    class2_dir = output_dir / "class2_dataset"
    no_class2_dir = output_dir / "no_class2_dataset"
    
    for dataset_dir in [class2_dir, no_class2_dir]:
        for split in ['train', 'val']:
            (dataset_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
            (dataset_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)
    
    # Get base directory from yaml file location
    yaml_dir = Path(data_yaml_path).parent
    
    # Process train and val splits
    for split in ['train', 'val']:
        if split in data_config:
            # Handle different path formats in YOLO yaml
            split_path = data_config[split]
            
            # Convert relative paths to absolute paths
            if not os.path.isabs(split_path):
                split_path = yaml_dir / split_path
            else:
                split_path = Path(split_path)
            
            # Your structure: data/train/images, data/val/images
            if 'images' in str(split_path):
                # Path already points to images directory (data/train/images)
                images_path = Path(split_path)
                labels_path = Path(str(split_path).replace('images', 'labels'))
            else:
                # Fallback: assume path points to parent directory
                images_path = split_path / 'images'
                labels_path = split_path / 'labels'
            
            print(f"🔍 Looking for {split} images in: {images_path}")
            print(f"🔍 Looking for {split} labels in: {labels_path}")
            
            if images_path.exists() and labels_path.exists():
                img_count_class2 = 0
                img_count_no_class2 = 0
                
                for img_file in images_path.glob('*'):
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp']:
                        label_file = labels_path / f"{img_file.stem}.txt"
                        
                        if has_class_2(label_file):
                            # Copy to class2 dataset
                            shutil.copy2(img_file, class2_dir / 'images' / split / img_file.name)
                            if label_file.exists():
                                shutil.copy2(label_file, class2_dir / 'labels' / split / label_file.name)
                            img_count_class2 += 1
                        else:
                            # Copy to no_class2 dataset
                            shutil.copy2(img_file, no_class2_dir / 'images' / split / img_file.name)
                            if label_file.exists():
                                shutil.copy2(label_file, no_class2_dir / 'labels' / split / label_file.name)
                            img_count_no_class2 += 1
                
                print(f"✅ {split}: {img_count_class2} images with class 2 (OxygenTank), {img_count_no_class2} images without class 2")
                
                # Check if we have a reasonable split
                if img_count_class2 == 0:
                    print(f"⚠️  No images with OxygenTank (class 2) found in {split} set")
                elif img_count_no_class2 == 0:
                    print(f"⚠️  All images in {split} set contain OxygenTank (class 2)")
            else:
                print(f"❌ Could not find {split} dataset at {images_path} or {labels_path}")
                print(f"   Images path exists: {images_path.exists()}")
                print(f"   Labels path exists: {labels_path.exists()}")
    
    # Check if we have enough data for both datasets
    class2_train_count = len(list((class2_dir / 'images' / 'train').glob('*')))
    no_class2_train_count = len(list((no_class2_dir / 'images' / 'train').glob('*')))
    
    print(f"📊 Dataset split summary:")
    print(f"   • Images with OxygenTank (class 2): {class2_train_count}")
    print(f"   • Images without OxygenTank: {no_class2_train_count}")
    
    if class2_train_count == 0 and no_class2_train_count == 0:
        raise Exception("No images found in either dataset split")
    elif no_class2_train_count == 0:
        raise Exception("All images contain OxygenTank (class 2) - cannot use class-specific training")
    elif class2_train_count == 0:
        raise Exception("No images with OxygenTank (class 2) found - class-specific training not needed")
    else:
        print(f"❌ Could not find {split} dataset at {images_path} or {labels_path}")
    
    # Create YAML files for both datasets
    class2_yaml = class2_dir / "data.yaml"
    no_class2_yaml = no_class2_dir / "data.yaml"
    
    # Class2 dataset config (minimal augmentation)
    class2_config = data_config.copy()
    class2_config['train'] = str(class2_dir / 'images' / 'train')
    class2_config['val'] = str(class2_dir / 'images' / 'val')
    
    with open(class2_yaml, 'w') as f:
        yaml.dump(class2_config, f)
    
    # No class2 dataset config (full augmentation)
    no_class2_config = data_config.copy()
    no_class2_config['train'] = str(no_class2_dir / 'images' / 'train')
    no_class2_config['val'] = str(no_class2_dir / 'images' / 'val')
    
    with open(no_class2_yaml, 'w') as f:
        yaml.dump(no_class2_config, f)
    
    print(f"✅ Created class-specific datasets in {output_dir}")
    return str(class2_yaml), str(no_class2_yaml)

def train_with_reduced_augmentation(args, this_dir):
    """Fallback training method with reduced augmentation overall"""
    
    print("🔄 Using fallback training with reduced augmentation...")
    
    # Load model
    model = YOLO(os.path.join(this_dir, "yolov8l.pt"))
    
    # Use moderate augmentation (between full and minimal)
    moderate_aug = {}
    for key, value in AUGMENTATION.items():
        if isinstance(value, (int, float)):
            moderate_aug[key] = value * 0.5  # Reduce all augmentation by 50%
        else:
            moderate_aug[key] = value
    
    # Train with reduced augmentation
    results = model.train(
        data=os.path.join(this_dir, "yolo_params.yaml"),
        epochs=args.epochs,
        device=0,
        single_cls=args.single_cls,
        mosaic=args.mosaic * 0.5,  # Reduced mosaic
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        imgsz=args.imgsz,
        batch=args.batch,
        project="trained_model_outputs",
        name="reduced_augmentation_training",
        verbose=True,
        patience=15,
        **moderate_aug
    )
    
    return results

def train_with_class_specific_augmentation(args, this_dir):
    """Train model with different augmentation strategies for different classes"""
    
    original_data_yaml = os.path.join(this_dir, "yolo_params.yaml")
    temp_datasets_dir = os.path.join(this_dir, "temp_class_datasets")
    
    try:
        # Create class-specific datasets
        class2_yaml, no_class2_yaml = create_class_specific_datasets(original_data_yaml, temp_datasets_dir)
        
        # Check if datasets were created successfully
        with open(no_class2_yaml, 'r') as f:
            no_class2_config = yaml.safe_load(f)
        
        # Verify that no_class2 dataset has images
        train_path = Path(no_class2_config['train'])
        if not any(train_path.glob('*')):
            print("⚠️  No images without class 2 found. Falling back to standard training with reduced augmentation.")
            return train_with_reduced_augmentation(args, this_dir)
        
        # Load base model
        model = YOLO(os.path.join(this_dir, "yolov8l.pt"))
        
        print("🚀 Starting training with class-specific augmentation...")
        
        # Phase 1: Train on images without class 2 with full augmentation
        print("📈 Phase 1: Training on non-class-2 images with full augmentation...")
        results1 = model.train(
            data=no_class2_yaml,
            epochs=int(args.epochs * 0.6),  # 60% of total epochs
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
            name="phase1_no_class2",
            verbose=True,
            patience=15,
            **AUGMENTATION
        )
        
    except Exception as e:
        print(f"❌ Error in dataset splitting: {e}")
        print("🔄 Falling back to standard training with reduced augmentation for class 2 protection...")
        return train_with_reduced_augmentation(args, this_dir)
    
    # Phase 2: Continue training with class 2 images using minimal augmentation
    print("📈 Phase 2: Fine-tuning with class-2 images using minimal augmentation...")
    
    # Load the best model from phase 1
    best_model_path = results1.save_dir / 'weights' / 'best.pt'
    model = YOLO(str(best_model_path))
    
    # Train on class 2 data with minimal augmentation
    results2 = model.train(
        data=class2_yaml,
        epochs=int(args.epochs * 0.2),  # 20% of total epochs
        device=0,
        single_cls=args.single_cls,
        mosaic=0.1,  # Reduced mosaic for class 2
        optimizer=args.optimizer,
        lr0=args.lr0 * 0.1,  # Reduced learning rate for fine-tuning
        lrf=args.lrf,
        momentum=args.momentum,
        imgsz=args.imgsz,
        batch=args.batch,
        project="trained_model_outputs",
        name="phase2_class2_minimal_aug",
        verbose=True,
        patience=10,
        **CLASS_2_AUGMENTATION
    )
    
    # Phase 3: Final training on complete dataset with balanced approach
    print("📈 Phase 3: Final training on complete dataset...")
    
    # Load the best model from phase 2
    best_model_path = results2.save_dir / 'weights' / 'best.pt'
    model = YOLO(str(best_model_path))
    
    # Final training with moderate augmentation
    moderate_aug = AUGMENTATION.copy()
    for key in moderate_aug:
        if isinstance(moderate_aug[key], (int, float)):
            moderate_aug[key] = moderate_aug[key] * 0.7  # Reduce augmentation by 30%
    
    results3 = model.train(
        data=original_data_yaml,
        epochs=int(args.epochs * 0.2),  # Remaining 20% of total epochs
        device=0,
        single_cls=args.single_cls,
        mosaic=args.mosaic * 0.7,
        optimizer=args.optimizer,
        lr0=args.lr0 * 0.05,  # Very low learning rate for final fine-tuning
        lrf=args.lrf,
        momentum=args.momentum,
        imgsz=args.imgsz,
        batch=args.batch,
        project="trained_model_outputs",
        name="phase3_final_balanced",
        verbose=True,
        patience=8,
        **moderate_aug
    )
    
    # Cleanup temporary datasets
    if os.path.exists(temp_datasets_dir):
        shutil.rmtree(temp_datasets_dir)
        print("🧹 Cleaned up temporary datasets")
    
    print("✅ Multi-phase training completed!")
    print(f"📊 Phase 1 Results: {results1}")
    print(f"📊 Phase 2 Results: {results2}")  
    print(f"📊 Phase 3 Results: {results3}")
    
    return results3

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train YOLO with class-specific data augmentation')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Total number of training epochs')
    parser.add_argument('--mosaic', type=float, default=MOSAIC, help='Mosaic augmentation probability')
    parser.add_argument('--optimizer', type=str, default=OPTIMIZER, help='Optimizer type')
    parser.add_argument('--momentum', type=float, default=MOMENTUM, help='Optimizer momentum')
    parser.add_argument('--lr0', type=float, default=LR0, help='Initial learning rate')
    parser.add_argument('--lrf', type=float, default=LRF, help='Final learning rate factor')
    parser.add_argument('--single_cls', type=bool, default=SINGLE_CLS, help='Single class training')
    parser.add_argument('--imgsz', type=int, default=IMG_SIZE, help='Input image size')
    parser.add_argument('--batch', type=int, default=BATCH, help='Batch size')
    parser.add_argument('--class2-protection', action='store_true', default=True, 
                       help='Enable class 2 protection (minimal augmentation)')
    parser.add_argument('--inspect', action='store_true', help='Inspect dataset structure and exit')
    
    args = parser.parse_args()
    
    # Ensure the script runs from its own directory
    this_dir = os.path.dirname(__file__)
    os.chdir(this_dir)
    
    # If inspect flag is set, just inspect and exit
    if args.inspect:
        inspect_dataset_structure(this_dir)
        exit(0)
    
    print("🎯 Starting YOLO training with class-2 protection...")
    print(f"📋 Configuration:")
    print(f"   • Total epochs: {args.epochs}")
    print(f"   • Batch size: {args.batch}")
    print(f"   • Image size: {args.imgsz}")
    print(f"   • Class 2 protection: {'✅ Enabled' if args.class2_protection else '❌ Disabled'}")
    
    # Inspect dataset structure first
    inspect_dataset_structure(this_dir)
    
    if args.class2_protection:
        results = train_with_class_specific_augmentation(args, this_dir)
    else:
        # Standard training without class-specific handling
        print("⚠️  Training without class 2 protection - using standard augmentation for all classes")
        model = YOLO(os.path.join(this_dir, "yolov8l.pt"))
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
            name="standard_training",
            verbose=True,
            patience=15,
            **AUGMENTATION
        )
    
    print("🎉 Training completed successfully!")
    print(f"📈 Final results: {results}")
"""
=============================================================================
2-STAGE SYSTEM: Vehicle Detection + Color Classification
=============================================================================
HOW IT WORKS:
  Stage 1 - Detection Model: Detects vehicles in the frame (car, truck, etc.)
            and draws bounding boxes around them.
  Stage 2 - Color Model:     Takes each cropped bounding box from Stage 1
            and classifies its color (red, blue, white, etc.)

OUTPUT: 2 separate model files:
  - runs/detect/visdrone_detection/weights/best.pt  (finds the vehicles)
  - runs/classify/vcor_color_classifier/weights/best.pt (identifies color)

PROS: High accuracy for each task.
CONS: Slower at inference (two models run per frame).
=============================================================================
"""

import os
from ultralytics import YOLO


def train_stage1_detection():
    """
    Stage 1: Train a YOLO detection model on VisDrone to detect vehicles.
    Produces: runs/detect/visdrone_detection/weights/best.pt
    """
    print("\n" + "="*60)
    print("STAGE 1: Training Vehicle Detection Model on VisDrone")
    print("="*60)

    model = YOLO("yolo26n.pt")  # Start from pre-trained YOLO26 nano weights

    model.train(
        data="VisDrone.yaml",   # VisDrone dataset config (10 classes)
        epochs=100,
        imgsz=640,
        batch=16,
        name="visdrone_detection",
        device=0,
        patience=20,            # Early stopping
        mosaic=1.0,             # Mosaic augmentation for small objects
        mixup=0.1,
        lr0=0.01,
    )

    print("\n✅ Stage 1 complete!")
    print("   Model saved: runs/detect/visdrone_detection/weights/best.pt")


def train_stage2_color():
    """
    Stage 2: Train a YOLO classification model on the 'vcor' dataset
    to identify vehicle colors.
    Requires Stage 1 to be done first so we can transfer detection weights.
    Produces: runs/classify/vcor_color_classifier/weights/best.pt
    """
    print("\n" + "="*60)
    print("STAGE 2: Training Vehicle Color Classification Model")
    print("="*60)

    vcor_data_path = "/home/giblon/object-detection/vcor"
    detection_weights = "/home/giblon/object-detection/runs/detect/visdrone_detection/weights/best.pt"

    if not os.path.exists(vcor_data_path):
        print(f"❌ vcor dataset not found at: {vcor_data_path}")
        return

    # Initialize a YOLO classification model
    # Transfer weights from the detection model for better feature extraction
    model = YOLO("yolo26n-cls.yaml")
    if os.path.exists(detection_weights):
        print(f"   Loading detection weights for transfer learning: {detection_weights}")
        model.load(detection_weights)
    else:
        print("   ⚠️  Detection weights not found, training from scratch.")

    model.train(
        data=vcor_data_path,
        epochs=150,
        imgsz=224,              # Small image size — we only see cropped vehicle
        batch=16,
        name="vcor_color_classifier",
        device=0,
        task="classify",
        hsv_s=0.5,              # Vary saturation to help with silver/grey/white confusion
        hsv_v=0.5,              # Vary brightness for different lighting conditions
        degrees=15.0,           # Slight rotation for robustness
        fliplr=0.5,             # Horizontal flip
    )

    print("\n✅ Stage 2 complete!")
    print("   Model saved: runs/classify/vcor_color_classifier/weights/best.pt")


def run_inference_demo(image_or_video_path):
    """
    Demo: Run both models together on an image or video.
    This shows how the 2-stage system works during inference.
    """
    import cv2
    from collections import defaultdict

    det_path = "/home/giblon/object-detection/runs/detect/visdrone_detection/weights/best.pt"
    cls_path = "/home/giblon/object-detection/runs/classify/vcor_color_classifier/weights/best.pt"

    if not os.path.exists(det_path) or not os.path.exists(cls_path):
        print("❌ Both models must be trained before running inference.")
        return

    det_model = YOLO(det_path)
    cls_model = YOLO(cls_path)

    print(f"\nRunning 2-stage inference on: {image_or_video_path}")

    results = det_model(image_or_video_path, stream=True, verbose=False)
    for det_result in results:
        if det_result.boxes is None:
            continue
        for box in det_result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            frame = det_result.orig_img
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            cls_result = cls_model(crop, verbose=False)[0]
            color = cls_result.names[cls_result.probs.top1]
            conf  = cls_result.probs.top1conf.item()
            cls_name = det_model.names[int(box.cls.item())]
            print(f"  Detected: {cls_name} | Color: {color} ({conf:.1%})")


if __name__ == "__main__":
    # -------------------------------------------------------
    # STEP 1: Train the detection model (finds vehicles)
    train_stage1_detection()

    # -------------------------------------------------------
    # STEP 2: Train the color classifier (identifies color)
    # (Runs after Stage 1 so it can use the detection weights)
    train_stage2_color()

    # -------------------------------------------------------
    # OPTIONAL: Run a quick inference demo after training
    # Uncomment and set your path below:
    # run_inference_demo("./dataset_video/dataset1.mp4")

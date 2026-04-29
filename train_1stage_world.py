"""
=============================================================================
1-STAGE SYSTEM: YOLO-World Vehicle + Color Detection
=============================================================================
HOW IT WORKS:
  A single YOLO-World model is trained where the class names describe BOTH
  the vehicle type AND its color. For example:
    - "white car", "red car", "black truck", "blue van"

  At inference, you simply run the single model and the output box label
  already contains the color + vehicle type together.

OUTPUT: 1 single model file:
  - runs/detect/visdrone_world_color/weights/best.pt

PROS: Fast inference (1 model), simple deployment, great for drone/edge devices.
CONS: Requires a dataset with color+type labels (not standard VisDrone).
     Can be tricky to get high accuracy on all color combinations.

IMPORTANT NOTE:
  Standard VisDrone only has generic classes (pedestrian, car, van, truck...).
  For true 1-stage color detection, you need a dataset where each bounding box
  is labeled as e.g. "red car" instead of just "car".
  
  However, YOLO-World supports ZERO-SHOT inference — meaning you can define
  custom text classes at inference time WITHOUT retraining, which is a huge
  advantage. See run_zeroshot_inference() below.
=============================================================================
"""

import os
from ultralytics import YOLO


# Define your color+vehicle class names here.
# These will be used for both training (if you have a matching dataset)
# and for zero-shot inference.
COLOR_VEHICLE_CLASSES = [
    "white car", "black car", "red car", "blue car", "silver car",
    "grey car", "yellow car", "green car", "orange car", "brown car",
    "white truck", "black truck", "white van", "black van",
    "white bus", "yellow bus",
]


def train_world_model():
    """
    Fine-tune a YOLO-World model on VisDrone.
    
    NOTE: This trains on standard VisDrone classes (car, truck, van, etc.).
    For full color-aware training, you would need to replace VisDrone.yaml
    with a custom dataset where labels include color (e.g. 'red car').
    
    Produces: runs/detect/visdrone_world_color/weights/best.pt
    """
    print("\n" + "="*60)
    print("1-STAGE: Training YOLO-World Model on VisDrone")
    print("="*60)

    # Load the YOLO-World small model (pre-trained on large open-vocab dataset)
    model = YOLO("yolov8s-world.pt")

    # Set custom classes for the model — tell it what to look for
    # Uncomment the line below if you have a color-labeled dataset:
    # model.set_classes(COLOR_VEHICLE_CLASSES)

    model.train(
        data="VisDrone.yaml",   # Replace with color-labeled dataset YAML if available
        epochs=100,
        imgsz=640,
        batch=16,
        name="visdrone_world_color",
        device=0,
        patience=20,
        mosaic=1.0,
        mixup=0.1,
        lr0=0.01,
    )

    print("\n✅ Training complete!")
    print("   Model saved: runs/detect/visdrone_world_color/weights/best.pt")


def run_zeroshot_inference(image_or_video_path):
    """
    ZERO-SHOT: Use a pre-trained YOLO-World model with custom color+vehicle
    text prompts WITHOUT any additional training.
    
    This is the fastest way to test the 1-stage approach right now!
    The model will try to find objects matching your text descriptions.
    """
    print("\n" + "="*60)
    print("1-STAGE ZERO-SHOT: Running YOLO-World with Color+Vehicle Prompts")
    print("="*60)

    # Load the base YOLO-World model (no custom training needed)
    model = YOLO("yolov8s-world.pt")

    # Tell the model what to look for — color + vehicle descriptions
    model.set_classes(COLOR_VEHICLE_CLASSES)

    print(f"Classes set: {COLOR_VEHICLE_CLASSES}")
    print(f"Running inference on: {image_or_video_path}\n")

    # Run inference
    results = model(image_or_video_path, conf=0.25, verbose=True)

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls_id = int(box.cls.item())
            label  = model.names[cls_id]
            conf   = box.conf.item()
            print(f"  Detected: '{label}'  (confidence: {conf:.1%})")


def run_trained_inference(image_or_video_path):
    """
    Run inference using the fine-tuned YOLO-World model after training.
    """
    model_path = "/home/giblon/object-detection/runs/detect/visdrone_world_color/weights/best.pt"

    if not os.path.exists(model_path):
        print(f"❌ Trained model not found at: {model_path}")
        print("   Please run train_world_model() first.")
        return

    print(f"\nRunning 1-stage inference on: {image_or_video_path}")
    model = YOLO(model_path)
    results = model(image_or_video_path, conf=0.25, verbose=True)

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls_id = int(box.cls.item())
            label  = model.names[cls_id]
            conf   = box.conf.item()
            print(f"  Detected: '{label}'  (confidence: {conf:.1%})")


if __name__ == "__main__":
    # -------------------------------------------------------
    # OPTION A: Try zero-shot inference RIGHT NOW (no training needed!)
    # Just set your image or video path and run this.
    # Uncomment below:
    # run_zeroshot_inference("./dataset_video/dataset1.mp4")

    # -------------------------------------------------------
    # OPTION B: Fine-tune YOLO-World on VisDrone (improves drone accuracy)
    # then run inference on a video.
    # Uncomment below:
    # train_world_model()
    # run_trained_inference("./dataset_video/dataset1.mp4")

    # -------------------------------------------------------
    # Default: Show the available options
    print("1-Stage YOLO-World System")
    print("-" * 40)
    print("A) Zero-shot inference (no training) — uncomment run_zeroshot_inference()")
    print("B) Train + inference                 — uncomment train_world_model() + run_trained_inference()")
    print("\nEdit the __main__ block to choose your option.")

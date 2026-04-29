import os
import cv2
import random
import glob
from collections import defaultdict
from ultralytics import YOLO

# =============================================================================
# CONFIGURATION & PATHS
# =============================================================================
# Detection / World Models
DET_MODEL_PATH = "runs/detect/visdrone_detection/weights/best.pt"
WORLD_MODEL_PATH = "runs/detect/visdrone_world_color/weights/best.pt"

# Classification Models (2-Stage)
CLS_MODEL_PATH = "runs/classify/vcor_color_classifier/weights/best.pt"

# Video to test on
TEST_VIDEOS = [
    "/home/giblon/object-detection/datasets/dataset_video/dataset1.mp4",
    "/home/giblon/object-detection/datasets/dataset_video/dataset2.mp4",
    "/home/giblon/object-detection/datasets/dataset_video/dataset3.mp4",
    "/home/giblon/object-detection/datasets/dataset_video/dataset4.mp4",
]

# Color Map for Visualization
COLOR_MAP = {
    "beige":  (180, 220, 240), "black":  (30,  30,  30),
    "blue":   (200, 80,  0),   "brown":  (30,  80,  130),
    "gold":   (0,   190, 240), "green":  (0,   180, 50),
    "grey":   (150, 150, 150), "orange": (0,   140, 255),
    "pink":   (190, 105, 255), "purple": (200, 0,   160),
    "red":    (0,   0,   220), "silver": (210, 210, 210),
    "tan":    (80,  140, 180), "white":  (245, 245, 245),
    "yellow": (0,   220, 220),
}
DEFAULT_BOX_COLOR = (0, 255, 0)

# YOLO-World Prompts
WORLD_CLASSES = [
    "white car", "black car", "red car", "blue car", "silver car",
    "grey car", "yellow car", "green car", "white truck", "black truck"
]

# =============================================================================
# 1. TRAINING FUNCTIONS
# =============================================================================

def train_detection_only():
    """Trains a standard vehicle detection model on VisDrone."""
    print("\n--- Starting Standard Detection Training ---")
    model = YOLO("yolo26n.pt")
    model.train(
        data="VisDrone.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        name="visdrone_detection",
        device=0
    )

def train_color_classifier():
    """Trains a color classification model using the vcor dataset."""
    print("\n--- Starting Color Classification Training ---")
    if not os.path.exists(DET_MODEL_PATH):
        print(f"❌ Error: Detection model not found at {DET_MODEL_PATH}. Train it first!")
        return
    model = YOLO("yolo26n-cls.yaml")
    model.load(DET_MODEL_PATH)
    model.train(
        data="/home/giblon/object-detection/vcor",
        epochs=150,
        imgsz=224,
        name="vcor_color_classifier",
        device=0
    )

def train_yolo_world():
    """Trains/Fine-tunes the YOLO-World model on VisDrone."""
    print("\n--- Starting YOLO-World Training ---")
    model = YOLO("yolov8s-world.pt")
    model.train(
        data="VisDrone.yaml",
        epochs=100,
        imgsz=640,
        name="visdrone_world_color",
        device=0
    )

def train_all():
    """Trains ALL models sequentially: Detection → Color Classifier → YOLO-World."""
    print("\n" + "="*60)
    print("TRAIN ALL: Running full training pipeline...")
    print("  Step 1/3: Standard Detection")
    print("  Step 2/3: Color Classifier (requires Step 1)")
    print("  Step 3/3: YOLO-World Fine-Tuning")
    print("="*60)
    print("⚠️  WARNING: This will take a very long time. Press Ctrl+C to cancel.\n")
    train_detection_only()
    train_color_classifier()
    train_yolo_world()
    print("\n✅ All training complete!")
    print("   Detection model : runs/detect/visdrone_detection/weights/best.pt")
    print("   Color classifier: runs/classify/vcor_color_classifier/weights/best.pt")
    print("   YOLO-World model: runs/detect/visdrone_world_color/weights/best.pt")

# =============================================================================
# 2. INFERENCE & TESTING FUNCTIONS
# =============================================================================

def run_2stage_video():
    """Run Detection + Classification on a video."""
    print("\n--- Running 2-Stage Inference Demo ---")
    if not os.path.exists(DET_MODEL_PATH) or not os.path.exists(CLS_MODEL_PATH):
        print("❌ Error: Missing models. Please train them first.")
        return

    det_model = YOLO(DET_MODEL_PATH)
    cls_model = YOLO(CLS_MODEL_PATH)

    for video_path in TEST_VIDEOS:
        if not os.path.exists(video_path): continue
        print(f"Processing: {video_path}")
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            results = det_model(frame, verbose=False)[0]
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0: continue
                
                # Classify Color
                cls_res = cls_model(crop, verbose=False)[0]
                color = cls_res.names[cls_res.probs.top1]
                
                # Draw
                cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_MAP.get(color, DEFAULT_BOX_COLOR), 2)
                cv2.putText(frame, color, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

            cv2.imshow("2-Stage Drone POV", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        cap.release()
    cv2.destroyAllWindows()

def run_world_zeroshot():
    """Runs YOLO-World with text prompts (No training needed)."""
    print("\n--- Running YOLO-World Zero-Shot Inference ---")
    model = YOLO("yolov8s-world.pt")
    model.set_classes(WORLD_CLASSES)

    for video_path in TEST_VIDEOS:
        if not os.path.exists(video_path): continue
        results = model.track(source=video_path, show=True, conf=0.25)

# =============================================================================
# 3. INTERACTIVE MENU
# =============================================================================

def menu():
    while True:
        print("\n" + "="*55)
        print("         DRONE OBJECT DETECTION MASTER MENU")
        print("="*55)
        print("--- TRAINING ---")
        print("[1] TRAIN: Standard Detection Model")
        print("[2] TRAIN: Color Classifier (requires Option 1 first)")
        print("[3] TRAIN: YOLO-World Fine-Tuning")
        print("[A] TRAIN: ⚡ ALL models sequentially (1 → 2 → 3)")
        print("-" * 55)
        print("--- TESTING ---")
        print("[4] TEST:  2-Stage System (Detection + Color)")
        print("[5] TEST:  1-Stage YOLO-World (Zero-Shot, no training)")
        print("[6] TEST:  1-Stage YOLO-World (Fine-tuned)")
        print("-" * 55)
        print("[Q] Quit")
        
        choice = input("\nSelect an option: ").strip().upper()
        
        if choice == '1':   train_detection_only()
        elif choice == '2': train_color_classifier()
        elif choice == '3': train_yolo_world()
        elif choice == 'A': train_all()
        elif choice == '4': run_2stage_video()
        elif choice == '5': run_world_zeroshot()
        elif choice == 'Q': break
        else: print("Invalid choice, please try again.")

if __name__ == "__main__":
    menu()

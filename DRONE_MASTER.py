import os
import cv2
import tkinter as tk
from tkinter import ttk
from ultralytics import YOLO

# =============================================================================
# CONFIGURATION & PATHS
# =============================================================================
# 2-Stage: Detection + Color Classifier paths
DET_MODEL_PATH  = "runs/detect/visdrone_detection/weights/best.pt"
CLS_MODEL_PATH  = "runs/classify/vcor_color_classifier/weights/best.pt"

# 1-Stage: YOLO-World fine-tuned path
WORLD_MODEL_PATH = "runs/detect/visdrone_world_color/weights/best.pt"

# Dataset paths
VCOR_DATA_PATH = "/home/giblon/object-detection/vcor"
TEST_VIDEOS = [
    "/home/giblon/object-detection/datasets/dataset_video/dataset1.mp4",
    "/home/giblon/object-detection/datasets/dataset_video/dataset2.mp4",
    "/home/giblon/object-detection/datasets/dataset_video/dataset3.mp4",
    "/home/giblon/object-detection/datasets/dataset_video/dataset4.mp4",
]

# Color Map for visualization (BGR format for OpenCV)
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

# YOLO-World: Color + Vehicle text prompts for zero-shot inference
# Expanded list from train_1stage_world.py for better coverage
WORLD_CLASSES = [
    "white car", "black car", "red car", "blue car", "silver car",
    "grey car", "yellow car", "green car", "orange car", "brown car",
    "white truck", "black truck", "white van", "black van",
    "white bus", "yellow bus",
]

# Plain vehicle classes for zero-shot detection (used with HSV color analysis)
# These are much more reliable than compound color+vehicle prompts
VEHICLE_CLASSES = ["car", "truck", "van", "bus", "vehicle"]

# All unique color names available for selection
ALL_COLORS = sorted(set(cls.split()[0] for cls in WORLD_CLASSES))


# =============================================================================
# UI HELPER: Color Selection Popup
# =============================================================================

def select_colors_popup():
    """
    Shows a tkinter checkbox popup so the pilot can choose which colors
    to detect on competition day.
    Returns a list of WORLD_CLASSES entries matching selected colors.
    Returns ALL WORLD_CLASSES if the pilot skips or cancels.
    """
    selected = []
    root = tk.Tk()
    root.title("Select Target Vehicle Colors")
    root.geometry("340x460")
    root.resizable(False, False)
    root.configure(bg="#1e1e2e")

    tk.Label(root, text="🚁  Competition Color Selector",
             font=("Arial", 14, "bold"), fg="#cdd6f4", bg="#1e1e2e").pack(pady=(18, 4))
    tk.Label(root, text="Select which vehicle colors to detect:",
             font=("Arial", 10), fg="#a6adc8", bg="#1e1e2e").pack(pady=(0, 12))

    frame = tk.Frame(root, bg="#1e1e2e")
    frame.pack(fill="both", expand=True, padx=24)

    vars_ = {}
    for color in ALL_COLORS:
        var = tk.BooleanVar(value=True)   # All checked by default
        vars_[color] = var
        box_color_hex = "#{:02x}{:02x}{:02x}".format(
            COLOR_MAP.get(color, DEFAULT_BOX_COLOR)[2],
            COLOR_MAP.get(color, DEFAULT_BOX_COLOR)[1],
            COLOR_MAP.get(color, DEFAULT_BOX_COLOR)[0]
        )
        cb = tk.Checkbutton(
            frame, text=f"  {color.capitalize()} vehicles",
            variable=var,
            font=("Arial", 11), fg="#cdd6f4", bg="#1e1e2e",
            selectcolor="#313244", activebackground="#1e1e2e",
            activeforeground="#cdd6f4",
            indicatoron=True,
        )
        cb.pack(anchor="w", pady=2)

    def confirm():
        for color, var in vars_.items():
            if var.get():
                # Include all WORLD_CLASSES entries that start with this color
                for cls in WORLD_CLASSES:
                    if cls.startswith(color) and cls not in selected:
                        selected.append(cls)
        root.destroy()

    def select_all():
        for var in vars_.values(): var.set(True)

    def clear_all():
        for var in vars_.values(): var.set(False)

    btn_frame = tk.Frame(root, bg="#1e1e2e")
    btn_frame.pack(pady=12)
    tk.Button(btn_frame, text="All",   command=select_all,  bg="#45475a", fg="#cdd6f4", width=7).pack(side="left",  padx=4)
    tk.Button(btn_frame, text="Clear", command=clear_all,   bg="#45475a", fg="#cdd6f4", width=7).pack(side="left",  padx=4)
    tk.Button(btn_frame, text="✅  Start", command=confirm, bg="#a6e3a1", fg="#1e1e2e", width=9,
              font=("Arial", 10, "bold")).pack(side="left", padx=4)

    root.mainloop()
    # Fallback: if nothing selected (pilot hit X), use all classes
    return selected if selected else WORLD_CLASSES


# =============================================================================
# DRAWING HELPER: Bounding Box with Dot + Dimensions
# =============================================================================

def draw_box_info(frame, x1, y1, x2, y2, label, box_color):
    """
    Draws:
      - Colored bounding box
      - Label with confidence above the box
      - Width x Height dimensions inside the box
      - Yellow center dot with (cx, cy) coordinate text
    """
    w = x2 - x1
    h = y2 - y1
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    # Bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)

    # Label above the box
    cv2.putText(frame, label,
                (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Width x Height inside the box (bottom-left corner)
    dim_text = f"{w}x{h}px"
    cv2.putText(frame, dim_text,
                (x1 + 3, y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1)

    # Center dot
    cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)  # Filled cyan dot
    cv2.circle(frame, (cx, cy), 5, (0, 0, 0), 1)        # Black outline

    # Center coordinate text
    coord_text = f"({cx},{cy})"
    cv2.putText(frame, coord_text,
                (cx + 7, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)


def classify_color_hsv(crop):
    """
    Classifies vehicle color using HSV pixel analysis on the CENTER of the crop.
    This is much more reliable than compound text prompts for zero-shot inference.
    Returns a color name string (e.g. 'red', 'white', 'black').
    """
    if crop is None or crop.size == 0:
        return "unknown"

    # Use the center 50% of the crop to avoid background edges
    h, w = crop.shape[:2]
    center = crop[h//4:3*h//4, w//4:3*w//4]
    if center.size == 0:
        center = crop

    hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
    avg_h = float(cv2.mean(hsv[:, :, 0])[0])  # Hue  (0-179)
    avg_s = float(cv2.mean(hsv[:, :, 1])[0])  # Sat  (0-255)
    avg_v = float(cv2.mean(hsv[:, :, 2])[0])  # Val  (0-255)

    # Low saturation = achromatic colors
    if avg_s < 40:
        if avg_v > 200:   return "white"
        elif avg_v > 140: return "silver"
        elif avg_v > 75:  return "grey"
        else:             return "black"

    # Chromatic colors — map hue ranges to color names
    if avg_h < 10 or avg_h > 165:  return "red"
    elif avg_h < 22:               return "orange"
    elif avg_h < 38:               return "yellow"
    elif avg_h < 85:               return "green"
    elif avg_h < 130:              return "blue"
    elif avg_h < 150:              return "purple"
    elif avg_h < 165:              return "pink"
    return "unknown"


# =============================================================================
# SECTION 1: 2-STAGE TRAINING (Detection → Color Classifier)
# =============================================================================
# HOW IT WORKS:
#   Stage 1 — Detection model detects vehicles and draws bounding boxes.
#   Stage 2 — Cropped boxes are passed to the color classifier.
# OUTPUT: 2 model files (best.pt for each stage)
# PROS: High accuracy for each task.
# CONS: Two models run per frame (slower on SBC).
# =============================================================================

def train_stage1_detection():
    """
    Stage 1 (2-Stage System): Train YOLO26n detection on VisDrone.
    Optimized for 100-120m drone altitude (tiny objects).
    Anti-overfitting: dropout, weight_decay, label_smoothing, close_mosaic.
    Produces: runs/detect/visdrone_detection/weights/best.pt
    """
    print("\n" + "="*60)
    print("STAGE 1: Training Vehicle Detection Model on VisDrone")
    print("         Optimized for 120m altitude + SBC deployment")
    print("="*60)

    model = YOLO("yolo26n.pt")  # YOLO26 Nano — fastest on SBC
    model.train(
        data="VisDrone.yaml",
        epochs=300,
        imgsz=1280,
        batch=4,
        patience=50,             # Early stopping — main overfitting guard
        # --- Augmentation (prevents overfitting by varying inputs) ---
        mosaic=1.0,
        close_mosaic=20,         # Disable mosaic last 20 epochs for stable convergence
        mixup=0.15,
        scale=0.9,
        fliplr=0.5,
        hsv_s=0.6,
        hsv_v=0.4,
        # --- Regularization (directly prevents overfitting) ---
        weight_decay=0.0005,     # L2 regularization
        dropout=0.1,             # Dropout in classifier head
        label_smoothing=0.05,    # Smooth hard labels slightly
        name="visdrone_detection",
        device=0,
    )
    print("\n✅ Stage 1 complete!")
    print(f"   Model saved: {DET_MODEL_PATH}")


def train_stage2_color():
    """
    Stage 2 (2-Stage System): Train YOLO26n-cls on vcor color dataset.
    Uses Stage 1 detection weights for transfer learning.
    imgsz=224 is correct here — model only sees cropped vehicle images.
    Anti-overfitting: dropout, weight_decay, label_smoothing.
    Produces: runs/classify/vcor_color_classifier/weights/best.pt
    """
    print("\n" + "="*60)
    print("STAGE 2: Training Vehicle Color Classification Model")
    print("         Using 224px — cropped vehicle images, SBC optimized")
    print("="*60)

    if not os.path.exists(VCOR_DATA_PATH):
        print(f"❌ vcor dataset not found at: {VCOR_DATA_PATH}")
        return

    model = YOLO("yolo26n-cls.yaml")

    det_weights = "/home/giblon/object-detection/" + DET_MODEL_PATH
    if os.path.exists(DET_MODEL_PATH):
        print(f"   Loading detection weights for transfer learning: {DET_MODEL_PATH}")
        model.load(DET_MODEL_PATH)
    elif os.path.exists(det_weights):
        print(f"   Loading detection weights for transfer learning: {det_weights}")
        model.load(det_weights)
    else:
        print("   ⚠️  Detection weights not found, training from scratch.")

    model.train(
        data=VCOR_DATA_PATH,
        epochs=200,
        imgsz=224,
        batch=32,
        patience=50,
        # --- Augmentation ---
        hsv_s=0.6,
        hsv_v=0.4,
        degrees=15.0,
        fliplr=0.5,
        # --- Regularization (prevents overfitting on 15-class color dataset) ---
        weight_decay=0.001,      # Slightly stronger L2 for small dataset
        dropout=0.2,             # Higher dropout — classification models overfit easily
        label_smoothing=0.1,     # Smooth labels (helps silver/grey/white confusion)
        name="vcor_color_classifier",
        device=0,
        task="classify",
    )
    print("\n✅ Stage 2 complete!")
    print(f"   Model saved: {CLS_MODEL_PATH}")


# =============================================================================
# SECTION 2: 1-STAGE TRAINING (YOLO-World)
# =============================================================================
# HOW IT WORKS:
#   A single YOLO-World model handles detection AND color identification.
#   Class names contain both type + color: e.g. "red car", "black truck".
#   Also supports ZERO-SHOT: no training needed at all using text prompts.
# OUTPUT: 1 single model file
# PROS: Fast inference, simple deployment, great for SBC/edge devices.
# CONS: Standard VisDrone doesn't have color labels — zero-shot compensates.
# =============================================================================

def train_yolo_world():
    """
    OPTION B — Fine-Tune YOLO-World with Color+Vehicle class names.
    Anti-overfitting: dropout, weight_decay, label_smoothing, close_mosaic.
    Produces: runs/detect/visdrone_world_color/weights/best.pt
    """
    print("\n" + "="*60)
    print("OPTION B: Fine-Tuning YOLO-World with Color+Vehicle Classes")
    print("          1 model that detects vehicles AND colors")
    print("          Optimized for 120m altitude + SBC deployment")
    print("="*60)

    model = YOLO("yolov8s-world.pt")
    model.set_classes(WORLD_CLASSES)
    print(f"   Training to detect: {WORLD_CLASSES}")

    model.train(
        data="VisDrone.yaml",
        epochs=300,
        imgsz=1280,
        batch=4,
        patience=50,
        # --- Augmentation ---
        mosaic=1.0,
        close_mosaic=20,         # Stable convergence at end of training
        mixup=0.15,
        scale=0.9,
        fliplr=0.5,
        hsv_s=0.6,
        hsv_v=0.4,
        lr0=0.01,
        # --- Regularization ---
        weight_decay=0.0005,     # L2 regularization
        dropout=0.1,             # Dropout in classifier head
        label_smoothing=0.05,    # Slight smoothing
        name="visdrone_world_color",
        device=0,
    )
    print("\n✅ YOLO-World (Option B) training complete!")
    print(f"   Model saved: {WORLD_MODEL_PATH}")
    print("   Use menu Option [6] to run this trained model.")


def train_all():
    """Trains ALL models sequentially: Stage1 Detection → Stage2 Color → YOLO-World."""
    print("\n" + "="*60)
    print("TRAIN ALL: Running full training pipeline...")
    print("  Step 1/3: Stage 1 — Detection (visdrone_detection)")
    print("  Step 2/3: Stage 2 — Color Classifier (vcor_color_classifier)")
    print("  Step 3/3: 1-Stage — YOLO-World (visdrone_world_color)")
    print("="*60)
    print("⚠️  WARNING: This will take many hours. Press Ctrl+C to cancel.\n")
    train_stage1_detection()
    train_stage2_color()
    train_yolo_world()
    print("\n✅ All training complete!")
    print(f"   Detection model : {DET_MODEL_PATH}")
    print(f"   Color classifier: {CLS_MODEL_PATH}")
    print(f"   YOLO-World model: {WORLD_MODEL_PATH}")


# =============================================================================
# SECTION 3: INFERENCE & TESTING FUNCTIONS
# =============================================================================

def run_2stage_video():
    """
    2-Stage inference: Detection model finds vehicles, then color classifier
    identifies the color of each detected vehicle.
    Pilot selects target colors via popup before video starts.
    """
    print("\n--- Running 2-Stage Inference (Detection + Color) ---")
    if not os.path.exists(DET_MODEL_PATH) or not os.path.exists(CLS_MODEL_PATH):
        print("❌ Error: Missing models. Please train Stage 1 and Stage 2 first.")
        return

    print("Opening color selection popup...")
    target_colors = set(cls.split()[0] for cls in select_colors_popup())
    print(f"Detecting colors: {target_colors}")

    det_model = YOLO(DET_MODEL_PATH)
    cls_model = YOLO(CLS_MODEL_PATH)

    for video_path in TEST_VIDEOS:
        if not os.path.exists(video_path):
            print(f"⚠️  Skipping (not found): {video_path}")
            continue
        print(f"Processing: {video_path}")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        delay = max(1, int(1000 / fps))   # Correct playback speed

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            results = det_model(frame, verbose=False)[0]
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0: continue

                cls_res    = cls_model(crop, verbose=False)[0]
                color      = cls_res.names[cls_res.probs.top1]
                conf       = cls_res.probs.top1conf.item()

                if color not in target_colors:
                    continue  # Skip colors not selected by pilot

                box_color = COLOR_MAP.get(color, DEFAULT_BOX_COLOR)
                label     = f"{color} {conf:.0%}"
                draw_box_info(frame, x1, y1, x2, y2, label, box_color)

            cv2.imshow("2-Stage Drone POV | Q=next video  ESC=quit", frame)
            key = cv2.waitKey(delay) & 0xFF
            if key == 27: break          # ESC = quit all
            if key == ord('q'): break    # Q = skip to next
        cap.release()
        cv2.destroyAllWindows()
        # Flush buffered key events so they don't carry into the next video
        for _ in range(10): cv2.waitKey(1)



def run_world_trained():
    """
    OPTION B — Run the fine-tuned YOLO-World model after training Option [3].
    Pilot selects target colors via popup before video starts.
    Bounding box colors, center dot, and dimensions shown for each detection.
    """
    print("\n" + "="*60)
    print("OPTION B: Trained YOLO-World Color+Vehicle Detection")
    print("          1 fine-tuned model for aerial drone POV")
    print("="*60)
    if not os.path.exists(WORLD_MODEL_PATH):
        print(f"❌ Trained model not found at: {WORLD_MODEL_PATH}")
        print("   Please run Option [3] to train first.")
        return

    print("Opening color selection popup...")
    target_colors = set(cls.split()[0] for cls in select_colors_popup())
    print(f"Detecting colors: {target_colors}")

    model = YOLO(WORLD_MODEL_PATH)
    for video_path in TEST_VIDEOS:
        if not os.path.exists(video_path):
            print(f"⚠️  Skipping (not found): {video_path}")
            continue
        print(f"Running on: {video_path}")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        delay = max(1, int(1000 / fps))
        win_name = f"Option B | {os.path.basename(video_path)} | Q=next  ESC=quit"

        stopped = False
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            results = model(frame, verbose=False)[0]
            for box in results.boxes:
                cls_id     = int(box.cls.item())
                label_text = model.names[cls_id]
                conf       = box.conf.item()
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                color_word = label_text.split()[0] if label_text.split() else "unknown"
                if color_word not in target_colors:
                    continue

                box_color = COLOR_MAP.get(color_word, DEFAULT_BOX_COLOR)
                draw_box_info(frame, x1, y1, x2, y2, f"{label_text} {conf:.0%}", box_color)

            cv2.imshow(win_name, frame)
            key = cv2.waitKey(delay) & 0xFF
            if key == 27:              # ESC = stop all videos
                stopped = True
                break
            elif key == ord('q'):      # Q = skip to next video only
                break
        cap.release()
        cv2.destroyAllWindows()
        # Flush buffered key events so they don't carry into the next video
        for _ in range(10): cv2.waitKey(1)
        if stopped:
            print("   ESC pressed — stopping all videos.")
            break


# =============================================================================
# INTERACTIVE MENU
# =============================================================================

def menu():
    while True:
        print("\n" + "="*65)
        print("           DRONE OBJECT DETECTION MASTER MENU")
        print("           Target: 100-120m altitude | SBC Deployment")
        print("="*65)
        print("--- TRAINING ---")
        print("[1] TRAIN: 2-Stage — Detection Model (Stage 1 of 2)")
        print("[2] TRAIN: 2-Stage — Color Classifier (Stage 2 of 2, needs [1])")
        print("[3] TRAIN: OPTION B — YOLO-World with Color+Vehicle Classes")
        print("[A] TRAIN: ⚡ ALL models sequentially (1 → 2 → 3)")
        print("-" * 65)
        print("--- TESTING (1 Single Model Option) ---")
        print("[6] TEST: Trained YOLO-World Color Detection (needs [3])")
        print("-" * 65)
        print("--- TESTING (2 Model System) ---")
        print("[4] TEST 2-Stage: Detection + Color Boxes (needs [1] & [2])")
        print("-" * 65)
        print("[Q] Quit")

        choice = input("\nSelect an option: ").strip().upper()

        if   choice == '1': train_stage1_detection()
        elif choice == '2': train_stage2_color()
        elif choice == '3': train_yolo_world()
        elif choice == 'A': train_all()
        elif choice == '4': run_2stage_video()
        elif choice == '6': run_world_trained()
        elif choice == 'Q':
            print("Exiting. Good luck with the competition! 🚁")
            break
        else:
            print("Invalid choice, please try again.")

if __name__ == "__main__":
    menu()

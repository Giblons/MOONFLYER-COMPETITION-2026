import os
import cv2
import tkinter as tk
from tkinter import ttk
from ultralytics import YOLO

# =============================================================================
# CONFIGURATION & PATHS
# =============================================================================
# 2-Stage: Detection + Color Classifier paths
DET_MODEL_PATH  = "models/visdrone_detection/weights/best.pt"
CLS_MODEL_PATH  = "models/vcor_color_classifier/weights/best.pt"

# Dataset paths
VCOR_DATA_PATH = "../vcor"
TEST_VIDEOS = [
    "../dataset_video/dataset1.mp4",
    "../dataset_video/dataset2.mp4",
    "../dataset_video/dataset3.mp4",
    "../dataset_video/dataset4.mp4",
]

# TRAIN_DATA_PATH = ["../TRAIN-SET/images/"]
# VAL_DATA_PATH = ["../VAL-SET/images/"]
# TEST_DATA_PATH = ["../TEST-SET/images/", "../TEST-SET-CHANLANGE/images/"]

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

ALL_COLORS = sorted(list(COLOR_MAP.keys()))

# =============================================================================
# UI HELPER: Color Selection Popup
# =============================================================================

def select_colors_popup():
    """
    Shows a tkinter checkbox popup so the pilot can choose which colors
    to detect on competition day.
    Returns a list of selected colors.
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
                selected.append(color)
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
    return selected if selected else ALL_COLORS


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


# =============================================================================
# SECTION 1: 2-STAGE TRAINING (Detection → Color Classifier)
# =============================================================================

def train_stage1_detection():
    """
    Stage 1 (2-Stage System): Train YOLOv26n detection on VisDrone.
    Produces: runs/detect/visdrone_detection/weights/best.pt
    """
    print("\n" + "="*60)
    print("STAGE 1: Training Vehicle Detection Model on VisDrone")
    print("         Using latest YOLOv26n")
    print("="*60)

    model = YOLO("yolo26n.pt")  # Updated to yolov26n
    model.train(
        data="Task1.yaml",
        classes=[0, 1, 2, 3, 4],
        epochs=100,
        imgsz=640,
        batch=4,
        patience=50,
        mosaic=1.0,
        close_mosaic=20,
        mixup=0.15,
        scale=0.9,
        fliplr=0.5,
        hsv_s=0.6,
        hsv_v=0.4,
        weight_decay=0.0005,
        dropout=0.1,
        label_smoothing=0.05,
        name="visdrone_detection",
        project="models",
        device=0,
        deterministic=False,
    )
    print("\n✅ Stage 1 complete!")
    print(f"   Model saved: {DET_MODEL_PATH}")


def train_stage2_color():
    """
    Stage 2 (2-Stage System): Train YOLOv26n-cls on vcor color dataset.
    Produces: runs/classify/vcor_color_classifier/weights/best.pt
    """
    print("\n" + "="*60)
    print("STAGE 2: Training Vehicle Color Classification Model")
    print("         Using latest YOLOv26n-cls")
    print("="*60)

    if not os.path.exists(VCOR_DATA_PATH):
        print(f"❌ vcor dataset not found at: {VCOR_DATA_PATH}")
        return

    model = YOLO("yolo26n.pt")  # Updated to yolov26n-cls

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
        hsv_s=0.6,
        hsv_v=0.4,
        degrees=15.0,
        fliplr=0.5,
        weight_decay=0.001,
        dropout=0.2,
        label_smoothing=0.1,
        name="vcor_color_classifier",
        project="models",
        device=0,
        task="classify",
        deterministic=False,
    )
    print("\n✅ Stage 2 complete!")
    print(f"   Model saved: {CLS_MODEL_PATH}")


def train_all():
    print("\n" + "="*60)
    print("TRAIN ALL: Running full 2-stage training pipeline...")
    print("  Step 1/2: Stage 1 — Detection (visdrone_detection)")
    print("  Step 2/2: Stage 2 — Color Classifier (vcor_color_classifier)")
    print("="*60)
    train_stage1_detection()
    train_stage2_color()
    print("\n✅ All training complete!")


# =============================================================================
# SECTION 2: 2-STAGE TESTING
# =============================================================================

def run_2stage_video():
    """
    2-Stage inference: Detection model finds vehicles, then color classifier
    identifies the color of each detected vehicle.
    """
    print("\n--- Running 2-Stage Inference (Detection + Color) ---")
    if not os.path.exists(DET_MODEL_PATH) or not os.path.exists(CLS_MODEL_PATH):
        print("❌ Error: Missing models. Please train Stage 1 and Stage 2 first.")
        return

    print("Opening color selection popup...")
    target_colors = set(select_colors_popup())
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
        delay = max(1, int(1000 / fps))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            results = det_model(frame, verbose=False)[0]
            for box in results.boxes:
                cls_id = int(box.cls[0])

                if cls_id not in [2, 3, 4, 5, 8, 9]:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                h, w = frame.shape[:2]
                if x1 <= 0 or y1 <= 0 or x2 >= w or y2 >= h:
                    continue

                crop_w, crop_h = x2 - x1, y2 - y1
                if crop_w < 32 or crop_h < 32:
                    continue

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0: continue

                cls_res = cls_model(crop, verbose=False)[0]
                color   = cls_res.names[cls_res.probs.top1]
                conf    = cls_res.probs.top1conf.item()

                if color not in target_colors:
                    continue

                box_color = COLOR_MAP.get(color, DEFAULT_BOX_COLOR)
                label     = f"{color} {conf:.0%}"
                draw_box_info(frame, x1, y1, x2, y2, label, box_color)

            cv2.imshow("2-Stage Drone POV | Q=next video  ESC=quit", frame)
            key = cv2.waitKey(delay) & 0xFF
            if key == 27: break
            if key == ord('q'): break
        cap.release()
        cv2.destroyAllWindows()
        for _ in range(10): cv2.waitKey(1)


# =============================================================================
# INTERACTIVE MENU
# =============================================================================

def menu():
    while True:
        print("\n" + "="*65)
        print("           VISDRONE 2-STAGE MODEL MENU (YOLOv26n)")
        print("="*65)
        print("--- TRAINING ---")
        print("[1] TRAIN: 2-Stage — Detection Model (Stage 1 of 2)")
        print("[2] TRAIN: 2-Stage — Color Classifier (Stage 2 of 2, needs [1])")
        print("[A] TRAIN: ⚡ ALL models sequentially (1 → 2)")
        print("-" * 65)
        print("--- TESTING ---")
        print("[T] TEST 2-Stage: Detection + Color Boxes (needs [1] & [2])")
        print("-" * 65)
        print("[Q] Quit")

        choice = input("\nSelect an option: ").strip().upper()

        if   choice == '1': train_stage1_detection()
        elif choice == '2': train_stage2_color()
        elif choice == 'A': train_all()
        elif choice == 'T': run_2stage_video()
        elif choice == 'Q':
            print("Exiting. Good luck with the competition! 🚁")
            break
        else:
            print("Invalid choice, please try again.")

if __name__ == "__main__":
    menu()

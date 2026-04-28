import os
import cv2
import random
from collections import defaultdict
from ultralytics import YOLO

# --- Same models as in test_vcor.py ---
MODEL1_PATH = "/home/giblon/object-detection/runs/classify/vcor_visdrone_detection_augmented-2/weights/best.pt"
MODEL2_PATH = "/home/giblon/object-detection/runs/classify/vcor_visdrone_world_augmented/weights/best.pt"

# --- Detection model to find vehicles in video frames ---
DET_MODEL_PATH = "/home/giblon/object-detection/runs/detect/visdrone_detection/weights/best.pt"

VIDEOS = [
        "./dataset_video/dataset1.mp4",
        "./dataset_video/dataset2.mp4",
        #"./dataset_video/dataset3.mp4",
        #"./dataset_video/dataset4.mp4"
]

# Color map for drawing bounding boxes per color (BGR format)
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
DEFAULT_COLOR = (0, 255, 0)


def process_video(video_path, det_model, cls_model, output_path, model_name="Model 1"):
    """
    Processes every frame of a video:
     - Detects vehicles
     - Classifies their color
     - Draws color-coded bounding boxes with labels
     - Saves the annotated video
     - Returns a color distribution dict
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ❌ Could not open video: {video_path}")
        return {}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS))

    print(f"  Video info: {total_frames} frames | {width}x{height} | {fps} FPS")
    print(f"  Saving output to: {output_path}")

    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    color_counts = defaultdict(int)
    total_vehicles = 0
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % 60 == 0:
            print(f"  Frame {frame_count}/{total_frames} ({frame_count/total_frames:.0%})...")

        # Run detection
        det_results = det_model(frame, verbose=False)[0]

        if det_results.boxes is not None:
            for box in det_results.boxes:
                cls_id     = int(box.cls.item())
                class_name = det_model.names[cls_id]
                conf_det   = box.conf.item()

                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Crop for color classification
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                # Run color classification
                cls_results = cls_model(crop, verbose=False)[0]
                color = cls_results.names[cls_results.probs.top1]
                conf_cls = cls_results.probs.top1conf.item()

                color_counts[color] += 1
                total_vehicles += 1

                # Pick box color based on predicted vehicle color
                box_color = COLOR_MAP.get(color, DEFAULT_COLOR)

                # Draw bounding box and center dot
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.circle(frame, (center_x, center_y), 3, box_color, -1)
                cv2.putText(frame, f"({center_x}, {center_y})", (center_x + 5, center_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1)

                # Draw label: "car | red (96%)"
                label = f"{class_name} | {color} ({conf_cls:.0%})"
                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                label_y = max(y1, text_h + 5)
                cv2.rectangle(frame, (x1, label_y - text_h - 5), (x1 + text_w, label_y + 2), box_color, -1)
                # Pick black or white text based on box brightness
                brightness = 0.114*box_color[0] + 0.587*box_color[1] + 0.299*box_color[2]
                text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
                cv2.putText(frame, label, (x1, label_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

        out.write(frame)

    cap.release()
    out.release()

    print(f"  ✅ Video saved to {output_path}")
    return color_counts, total_vehicles


def print_stats(color_counts, total_vehicles, model_name):
    if total_vehicles == 0:
        print(f"  ⚠️  No vehicles detected.")
        return

    print(f"\n  [{model_name}] Color Distribution ({total_vehicles} total vehicle detections):")
    print(f"  {'Color':<12} {'Count':>6}  {'Percentage':>10}  Chart")
    print(f"  {'-'*50}")
    for color, count in sorted(color_counts.items(), key=lambda x: -x[1]):
        pct = count / total_vehicles * 100
        bar = "█" * int(pct / 3)
        print(f"  {color:<12} {count:>6}  {pct:>8.1f}%   {bar}")
    print()


def run_all():
    print("Loading models...")

    for path, name in [(MODEL1_PATH, "Model 1"), (MODEL2_PATH, "Model 2"), (DET_MODEL_PATH, "Detection")]:
        if not os.path.exists(path):
            print(f"❌ {name} not found: {path}")
            return

    det_model = YOLO(DET_MODEL_PATH)
    model1    = YOLO(MODEL1_PATH)
    model2    = YOLO(MODEL2_PATH)

    print("✅ Models loaded.\n")

    for vid_path in VIDEOS:
        dataset_name = os.path.splitext(os.path.basename(vid_path))[0]
        print(f"\n{'='*55}")
        print(f"  Testing: {dataset_name}  ({vid_path})")
        print(f"{'='*55}")

        if not os.path.exists(vid_path):
            print(f"  ❌ File not found, skipping.\n")
            continue

        # --- Model 1 ---
        print(f"\n  --- Model 1 (VisDrone Detection Base, Augmented) ---")
        out1 = f"output_{dataset_name}_model1.mp4"
        color_counts1, total1 = process_video(vid_path, det_model, model1, out1, "Model 1")
        print_stats(color_counts1, total1, "Model 1")

        # --- Model 2 ---
        print(f"  --- Model 2 (VisDrone World Base, Augmented) ---")
        out2 = f"output_{dataset_name}_model2.mp4"
        color_counts2, total2 = process_video(vid_path, det_model, model2, out2, "Model 2")
        print_stats(color_counts2, total2, "Model 2")


if __name__ == "__main__":
    run_all()

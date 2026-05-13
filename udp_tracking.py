import os
import cv2
import socket
import struct
import tkinter as tk
from tkinter import messagebox
from ultralytics import YOLO
from fastcrc import crc8

# --- CONFIGURATION ---
# Set USE_CAMERA to True to use webcam/camera index, False to use video files
USE_CAMERA = True
CAMERA_INDEX = 0

# Updated paths to reflect current user environment
DET_MODEL_PATH = "/home/rizky/object-detection/runs/detect/visdrone_detection/weights/best.pt"
CLS_MODEL_PATH = "/home/rizky/object-detection/runs/classify/vcor_visdrone_detection_augmented-2/weights/best.pt"

TEST_VIDEOS = [
    "/home/rizky/Downloads/dataset1.mp4",
    "/home/rizky/Downloads/dataset2.mp4",
    "/home/rizky/Downloads/dataset3.mp4",
]

# Color map for drawing (BGR format)
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

def select_colors_popup():
    """
    Creates a simple Tkinter popup to select colors to detect.
    Returns a list of selected colors.
    """
    root = tk.Tk()
    root.title("Select Colors to Track")
    
    selected_colors = []
    vars = {}

    tk.Label(root, text="Select target vehicle colors:", font=('Arial', 12, 'bold')).pack(pady=10)

    # Grid for checkboxes
    frame = tk.Frame(root)
    frame.pack(padx=20, pady=10)

    colors = sorted(COLOR_MAP.keys())
    for i, color in enumerate(colors):
        var = tk.BooleanVar()
        vars[color] = var
        cb = tk.Checkbutton(frame, text=color, variable=var, anchor='w')
        cb.grid(row=i // 3, column=i % 3, sticky='w', padx=5, pady=2)

    def on_submit():
        for color, var in vars.items():
            if var.get():
                selected_colors.append(color)
        if not selected_colors:
            messagebox.showwarning("Warning", "Please select at least one color.")
        else:
            root.destroy()

    tk.Button(root, text="Start Detection", command=on_submit, bg="#4CAF50", fg="white", font=('Arial', 10, 'bold')).pack(pady=20)
    
    # Handle window close
    root.protocol("WM_DELETE_WINDOW", lambda: root.destroy())
    
    root.mainloop()
    return selected_colors

def draw_box_info(frame, x1, y1, x2, y2, label, box_color):
    """
    Draws bounding box and label with background.
    """
    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
    
    # Label background
    cv2.rectangle(frame, (x1, y1 - 25), (x1 + w, y1), box_color, -1)
    
    # Pick black or white text based on box brightness
    brightness = 0.114*box_color[0] + 0.587*box_color[1] + 0.299*box_color[2]
    text_color = (0, 0, 0) if brightness > 128 else (255, 255, 255)
    
    cv2.putText(frame, label, (x1, y1 - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)

def run_2stage_video():
    """
    2-Stage inference: Detection model finds vehicles, then color classifier
    identifies the color of each detected vehicle, and sends tracking data via UDP.
    """
    print("\n--- Running 2-Stage Inference (Detection + Color) ---")
    if not os.path.exists(DET_MODEL_PATH) or not os.path.exists(CLS_MODEL_PATH):
        print(f"❌ Error: Missing models.")
        print(f"   DET: {DET_MODEL_PATH}")
        print(f"   CLS: {CLS_MODEL_PATH}")
        return

    print("Opening color selection popup...")
    target_colors = set(select_colors_popup())
    if not target_colors:
        print("No colors selected or window closed. Exiting.")
        return
        
    print(f"Detecting colors: {target_colors}")

    det_model = YOLO(DET_MODEL_PATH)
    cls_model = YOLO(CLS_MODEL_PATH)

    # --- UDP Socket Setup ---
    UDP_IP = "127.0.0.1"
    UDP_PORT = 13131
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    packet_header = ord('\x7E')
    packet_len = struct.calcsize('!BBBIIIIII')
    packet_type = ord('\x02')
    # ------------------------

    sources = [CAMERA_INDEX] if USE_CAMERA else TEST_VIDEOS
    for video_source in sources:
        if not USE_CAMERA and not os.path.exists(video_source):
            print(f"⚠️  Skipping (not found): {video_source}")
            continue
        
        source_name = f"Camera {video_source}" if USE_CAMERA else video_source
        print(f"Processing: {source_name}")
        
        cap = cv2.VideoCapture(video_source)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        delay = max(1, int(1000 / fps))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            results = det_model(frame, verbose=False)[0]
            for box in results.boxes:
                cls_id = int(box.cls[0])

                # Common vehicle classes in VisDrone/COCO: 2(car), 3(motorcycle), 4(airplane), 5(bus), 7(truck), etc.
                # Adjust based on your model's specific class mapping
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

                # --- SEND TRACKING DATA VIA UDP ---
                res_x = w
                res_y = h
                tgt_x = (x1 + x2) // 2  # Center X
                tgt_y = (y1 + y2) // 2  # Center Y
                tgt_w = crop_w          # Width
                tgt_h = crop_h          # Height

                payload = struct.pack('!BBBIIIIII', packet_header, packet_len, packet_type, res_x, res_y, tgt_x, tgt_y, tgt_w, tgt_h)
                crc = crc8.dvb_s2(payload)
                data = payload + struct.pack('!B', crc)
                sock.sendto(data, (UDP_IP, UDP_PORT))
                # ----------------------------------

            win_title = f"2-Stage Drone POV ({source_name}) | Q=next  ESC=quit"
            cv2.imshow(win_title, frame)
            key = cv2.waitKey(delay) & 0xFF
            if key == 27: break
            if key == ord('q'): break
            
        cap.release()
        cv2.destroyAllWindows()
        for _ in range(10): cv2.waitKey(1)

if __name__ == "__main__":
    run_2stage_video()
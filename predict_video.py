import cv2
import os
from ultralytics import YOLO

def process_video(input_path, output_path):
    print(f"\n--- Loading models for Video Processing ---")
    # Detection model
    det_model = YOLO("/home/giblon/object-detection/runs/detect/visdrone_detection/weights/best.pt")
    
    # Classification model (augmented weights)
    cls_model = YOLO("/home/giblon/object-detection/runs/classify/vcor_visdrone_detection_augmented-2/weights/best.pt")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"❌ Error: Could not open video {input_path}")
        return

    # Get video properties
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Setup VideoWriter to save the result
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Processing video: {input_path}")
    print(f"Total Frames: {total_frames} | Resolution: {width}x{height} | FPS: {fps}")

    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        
        # Print progress every 30 frames
        if frame_count % 30 == 0:
            print(f"Processing frame {frame_count}/{total_frames} ({(frame_count/total_frames):.1%})...")

        # 1. Run object detection tracking on the frame
        det_results = det_model.track(frame, persist=True, verbose=False, tracker="botsort.yaml")[0]

        # 2. Iterate through detected bounding boxes
        if det_results.boxes is not None:
            for box in det_results.boxes:
                cls_id = int(box.cls.item())
                class_name = det_model.names[cls_id]
                
                # Check if it has a tracking ID
                track_id = int(box.id.item()) if box.id is not None else -1

                # Get coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # 3. Crop the vehicle for color classification
                crop_img = frame[y1:y2, x1:x2]
                if crop_img.size == 0:
                    continue

                # 4. Run color classification on the cropped vehicle
                cls_results = cls_model(crop_img, verbose=False)[0]
                color_name = cls_results.names[cls_results.probs.top1]
                color_conf = cls_results.probs.top1conf.item()

                # 5. Draw bounding box and label
                # Label format: ID: Class | Color (Confidence%)
                if track_id != -1:
                    label = f"ID:{track_id} {class_name} | {color_name} ({color_conf:.0%})"
                else:
                    label = f"{class_name} | {color_name} ({color_conf:.0%})"
                
                # Draw Box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw Label Background
                (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(frame, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
                
                # Draw Text
                cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Write the processed frame to output video
        out.write(frame)

    cap.release()
    out.release()
    print(f"✅ Finished processing! Saved to {output_path}\n")

if __name__ == '__main__':
    # List of videos to process
    videos = [
        "/home/giblon/Downloads/dataset1.mp4",
        "/home/giblon/Downloads/dataset2.mp4",
        "/home/giblon/Downloads/dataset3.mp4",
        "/home/giblon/Downloads/dataset4.mp4"
    ]
    
    # Process each video sequentially
    for idx, vid_path in enumerate(videos):
        if os.path.exists(vid_path):
            output_file = f"output_dataset{idx+1}.mp4"
            process_video(vid_path, output_file)
        else:
            print(f"❌ Video not found: {vid_path}")

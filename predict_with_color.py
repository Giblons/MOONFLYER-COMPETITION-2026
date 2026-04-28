import cv2
import os
from ultralytics import YOLO

def predict_and_draw(source_path, output_path="output.jpg"):
    """
    Takes an image, finds vehicles, crops them, predicts their color, 
    and draws bounding boxes with both labels!
    """
    print(f"Loading models...")
    # Your detection model (finds the cars)
    det_model = YOLO("/home/giblon/object-detection/runs/detect/visdrone_detection/weights/best.pt")
    
    # Your classification model (finds the color)
    cls_model = YOLO("/home/giblon/object-detection/runs/classify/vcor_visdrone_detection_augmented-2/weights/best.pt")

    print(f"Reading image: {source_path}")
    img = cv2.imread(source_path)
    if img is None:
        print("Could not read image. Please check the path!")
        return

    # 1. Run object detection
    print("Running detection...")
    det_results = det_model(img, verbose=False)[0]

    # 2. Iterate through detected bounding boxes
    boxes_drawn = 0
    for box in det_results.boxes:
        cls_id = int(box.cls.item())
        class_name = det_model.names[cls_id]

        # Get coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        # 3. Crop the vehicle
        crop_img = img[y1:y2, x1:x2]
        if crop_img.size == 0:
            continue

        # 4. Run color classification on the cropped vehicle
        cls_results = cls_model(crop_img, verbose=False)[0]
        color_name = cls_results.names[cls_results.probs.top1]
        color_conf = cls_results.probs.top1conf.item()

        # 5. Draw everything!
        label = f"{class_name} | {color_name} ({color_conf:.1%})"
        
        # Bounding box (Green)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Label background
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
        
        # Label text (Black)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        boxes_drawn += 1

    # Save the final image
    cv2.imwrite(output_path, img)
    print(f"Success! Detected and colored {boxes_drawn} vehicles. Saved to: {output_path}")

if __name__ == '__main__':
    # Change this to any image path you want to test! 
    # Testing with an image from the vcor dataset as requested
    sample_img = "/home/giblon/object-detection/vcor/test/red/044ed301f3.jpg"
    
    if os.path.exists(sample_img):
        predict_and_draw(sample_img, "test_output_with_color.jpg")
    else:
        print("Please open predict_with_color.py and put the path to a real image on line 52!")

import os
from ultralytics import YOLO

def test_on_dataset():
    """
    Run formal validation on the 'test' split of the vcor dataset
    to get the final accuracy metrics.
    """
    print("Evaluating Model 1 (VisDrone Detection Base) on Test Set...")
    model1_path = "/home/giblon/object-detection/runs/classify/vcor_visdrone_detection_augmented-2/weights/best.pt"
    
    if os.path.exists(model1_path):
        model1 = YOLO(model1_path)
        metrics1 = model1.val(data="/home/giblon/object-detection/vcor", split="test", imgsz=224)
        print(f"Model 1 Test Accuracy (Top 1): {metrics1.top1:.4f}")
    else:
        print(f"Model 1 not found at {model1_path}")

    print("\nEvaluating Model 2 (VisDrone World Base) on Test Set...")
    model2_path = "/home/giblon/object-detection/runs/classify/vcor_visdrone_world_augmented/weights/best.pt"
    
    if os.path.exists(model2_path):
        model2 = YOLO(model2_path)
        metrics2 = model2.val(data="/home/giblon/object-detection/vcor", split="test", imgsz=224)
        print(f"Model 2 Test Accuracy (Top 1): {metrics2.top1:.4f}")
    else:
        print(f"Model 2 not found at {model2_path}")

import random
import glob

def test_each_color():
    """
    Picks one random image from every color folder in the test set
    and prints the model's prediction vs the actual color.
    """
    print("\n--- 2. Testing one image from EVERY color ---")
    model1_path = "/home/giblon/object-detection/runs/classify/vcor_visdrone_detection_augmented-2/weights/best.pt"
    test_dir = "/home/giblon/object-detection/vcor/test"
    
    if not os.path.exists(model1_path):
        print("Model not found.")
        return

    model = YOLO(model1_path)
    
    # Iterate through each color folder
    for color_folder in sorted(os.listdir(test_dir)):
        folder_path = os.path.join(test_dir, color_folder)
        if os.path.isdir(folder_path):
            # Get all images in this folder
            images = glob.glob(os.path.join(folder_path, "*.jpg"))
            if images:
                # Pick a random image
                random_image = random.choice(images)
                
                # Run inference quietly
                results = model(random_image, verbose=False)
                
                for r in results:
                    predicted_color = r.names[r.probs.top1]
                    confidence = r.probs.top1conf.item()
                    
                    # Check if correct
                    status = "✅" if predicted_color == color_folder else "❌"
                    
                    print(f"Actual: {color_folder:<10} | Predicted: {predicted_color:<10} {status} (Confidence: {confidence:.2%})")

if __name__ == "__main__":
    print("--- 1. Testing full dataset ---")
    test_on_dataset()
    
    test_each_color()

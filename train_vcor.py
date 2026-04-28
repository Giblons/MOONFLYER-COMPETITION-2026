import os
from ultralytics import YOLO

def train_vehicle_color():
    """
    Train classification models for vehicle color using the 'vcor' dataset.
    The models will start from the pre-trained detection weights.
    """
    print("Starting Vehicle Color Classification Training...")

    data_path = "/home/giblon/object-detection/vcor"
    
    # Model paths specified from test_models.py
    model1_path = "/home/giblon/object-detection/runs/detect/visdrone_detection/weights/best.pt"
    model2_path = "/home/giblon/object-detection/runs/detect/visdrone_world/weights/best.pt"

    # --- Train Model 1 ---
    print(f"\n{'='*50}")
    print(f"Training Model 1: {model1_path}")
    print(f"{'='*50}")
    
    if not os.path.exists(model1_path):
        print(f"Warning: {model1_path} not found. Skipping...")
    else:
        # Initialize a classification architecture and transfer weights from detection
        model1 = YOLO('yolo26n-cls.yaml')
        model1.load(model1_path)
        
        # Train on the classification dataset (Augmented for difficult colors)
        model1.train(
            data=data_path,
            epochs=150,  # Increased epochs
            imgsz=224,
            batch=16,
            name='vcor_visdrone_detection_augmented',
            device=0,
            task='classify',
            hsv_s=0.5, # Adjust saturation by 50% randomly
            hsv_v=0.5, # Adjust brightness by 50% randomly
            degrees=15.0, # Slight rotations
            fliplr=0.5 # Flip images left/right
        )
        print("Model 1 training completed.")

    # --- Train Model 2 ---
    print(f"\n{'='*50}")
    print(f"Training Model 2: {model2_path}")
    print(f"{'='*50}")
    
    if not os.path.exists(model2_path):
        print(f"Warning: {model2_path} not found. Skipping...")
    else:
        # Initialize a classification architecture and transfer weights from detection
        model2 = YOLO('yolo26n-cls.yaml')
        model2.load(model2_path)
        
        # Train on the classification dataset (Augmented for difficult colors)
        model2.train(
            data=data_path,
            epochs=150,  # Increased epochs
            imgsz=224,
            batch=16,
            name='vcor_visdrone_world_augmented',
            device=0,
            task='classify',
            hsv_s=0.5, # Adjust saturation by 50% randomly
            hsv_v=0.5, # Adjust brightness by 50% randomly
            degrees=15.0, # Slight rotations
            fliplr=0.5 # Flip images left/right
        )
        print("Model 2 training completed.")

if __name__ == '__main__':
    train_vehicle_color()

import os
from ultralytics import YOLO

def train_world_model():
    """
    Train a YOLO26-World model.
    YOLO-World is an open-vocabulary detection model that can be fine-tuned 
    on standard detection datasets like VisDrone.
    """
    print("Starting YOLO26-World Training...")
    # Load the YOLO26-World small model
    model = YOLO('yolov8s-world.pt')
    
    # Train the model on the VisDrone dataset
    results = model.train(
        data='VisDrone.yaml',  # Standard VisDrone dataset configuration
        epochs=100,
        imgsz=640,             # Kept at 640 for faster inference on Raspberry Pi
        batch=16,              # Increased batch size back up
        name='visdrone_world',
        device=0,              # Explicitly enable CUDA GPU
        patience=20,           # Early stopping to prevent overfitting
        mosaic=1.0,            # High mosaic for small object detection
        mixup=0.1,             # Mixup augmentation
        lr0=0.01               # Initial learning rate
    )
    return model

def detection():
    print("Starting YOLO Detection...")
    
    # Load an official or custom model
    model = YOLO("yolo26n.pt")  # Load an official Detect model
    
    results = model.train(
        data='VisDrone.yaml', 
        epochs=100, 
        imgsz=640,             # Kept at 640 for SBC compatibility
        batch=16, 
        name='visdrone_detection', 
        device=0,
        patience=20,
        mosaic=1.0,
        mixup=0.1,
        lr0=0.01
    )

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
    # Uncomment the function you want to run. 
    # Warning: Running them all at once will take a significant amount of time and compute resources.
    
    # 2. Train YOLO-WorldV2 Model
    train_world_model()
    

    # 4. Detection Model Training
    detection()
    
    # 5. Train Vehicle Color Classification
    train_vehicle_color()
    
    print("Script is ready. Please uncomment the function you wish to execute in the __main__ block.")

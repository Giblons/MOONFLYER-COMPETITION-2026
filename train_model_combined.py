import os
from ultralytics import YOLO

def train_obb_model():
    """
    Train an Oriented Bounding Box (OBB) model using YOLO26.
    Note that the standard VisDrone dataset only has horizontal bounding boxes (HBB). 
    To train an OBB model, your VisDrone dataset must be converted to the YOLO OBB format 
    (class x1 y1 x2 y2 x3 y3 x4 y4). OBB angles are constrained to 0–90 degrees.
    """
    print("Starting YOLO OBB Training...")
    # Load the OBB model
    model = YOLO(model="yolov26n-obb.yaml")
    model = YOLO('yolo26n-obb.pt')
    model = YOLO('yolo26n-obb.yaml').load('yolo26n-obb.pt')
      # Load from checkpoint

    
    # Train the model on the VisDrone dataset
    # Make sure you have an OBB compatible VisDrone-OBB.yaml config
    results = model.train(
        data='dota8.yaml',  # Dataset configuration file
        epochs=50,                 # Number of training epochs
        imgsz=640,                 # Image size
        batch=16,              # Batch size
        name='visdrone_obb',   # Experiment name
        device=0               # Explicitly enable CUDA GPU
    )
    return model

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

def setup_and_run_tracking():
    """
    Tracking is an inference task rather than a distinct model architecture in YOLO.
    It uses a trained object detection model (like the ones trained above or a standard yolov8n.pt) 
    and applies a tracker (e.g., BoT-SORT or ByteTrack) to associate objects across video frames.
    
    If you want to track on the VisDrone-MOT (Multi-Object Tracking) dataset or a video, 
    you do it during the `predict` or `track` phase.
    """
    print("Setting up YOLO Tracking...")
    
    # Load an official or custom model
    model = YOLO("yolo26n.pt")  # Load an official Detect model
    model = YOLO("yolo26n-seg.pt")  # Load an official Segment model
    model = YOLO("yolo26n-pose.pt")  # Load an official Pose model
    model = YOLO("path/to/best.pt")  # Load a custom-trained model
    
    try:
        # Perform tracking with the model on CUDA GPU
        #results = model.track("https://youtu.be/LNwODJXcvt4", show=True, device=0)  # Tracking with default tracker
        results = model.track("https://youtu.be/LNwODJXcvt4", show=True, tracker="bytetrack.yaml", device=0)  # with ByteTrack
    except Exception as e:
        print("Note: Provide a valid video path to see tracking in action.")

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
    
    # 1. Train OBB Model
    #train_obb_model()
    
    # 2. Train YOLO-WorldV2 Model
    train_world_model()
    
    # 3. Setup and Run Tracking
    #setup_and_run_tracking()

    # 4. Detection Model Training
    detection()
    
    # 5. Train Vehicle Color Classification
    train_vehicle_color()
    
    print("Script is ready. Please uncomment the function you wish to execute in the __main__ block.")

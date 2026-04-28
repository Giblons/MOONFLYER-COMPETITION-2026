from ultralytics import YOLO

def train_obb_model():
    """
    Train an Oriented Bounding Box (OBB) model using YOLO26.
    """
    print("Starting YOLO OBB Training...")
    # Load the OBB model in PyTorch format
    model = YOLO('yolo26n-obb.yaml').load('yolo26n-obb.pt')
    
    # Train the model on the VisDrone dataset
    results = model.train(
        data='dota8.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        name='visdrone_obb',
        device=0
    )
    
    # 1. Export the trained model to NCNN format
    print("Exporting trained model to NCNN...")
    model.export(format="ncnn")
    
    # 2. Load the newly exported NCNN model
    print("Loading NCNN model...")
    ncnn_model = YOLO("runs/obb/visdrone_obb/weights/best_ncnn_model")
    
    return ncnn_model

def train_world_model():
    """
    Train a YOLO26-World model.
    """
    print("Starting YOLO26-World Training...")
    # Load the YOLO26-World model in PyTorch format
    model = YOLO('yolov8s-world.pt')
    
    # Train the model
    results = model.train(
        data='VisDrone.yaml',
        epochs=100,
        imgsz=640,
        batch=16,
        name='visdrone_world',
        device=0
    )
    
    # 1. Export the trained model to NCNN format
    print("Exporting trained model to NCNN...")
    model.export(format="ncnn")
    
    # 2. Load the newly exported NCNN model
    print("Loading NCNN model...")
    ncnn_model = YOLO("runs/detect/visdrone_world/weights/best_ncnn_model")
    
    return ncnn_model

def setup_and_run_tracking():
    """
    Load a PyTorch model, export to NCNN, and run tracking.
    """
    print("Setting up YOLO Tracking with NCNN...")
    
    # Load an official Detect model in PyTorch format
    model = YOLO("yolo26n.pt")
    
    # 1. Export the model to NCNN format
    print("Exporting base model to NCNN...")
    model.export(format="ncnn")
    
    # 2. Load the newly exported NCNN model
    print("Loading NCNN model...")
    ncnn_model = YOLO("yolo26n_ncnn_model")
    
    try:
        # Perform tracking with the NCNN model
        results = ncnn_model.track("https://youtu.be/LNwODJXcvt4", show=True)
    except Exception as e:
        print("Note: Provide a valid video path to see tracking in action.")

def detection():
    print("Starting YOLO Detection...")
    
    # Load PyTorch model
    model = YOLO("yolo26n.pt")
    
    # Train the model
    results = model.train(data='VisDrone.yaml', epochs=100, imgsz=640, batch=16, name='visdrone_detection', device=0)
    
    # 1. Export the trained model to NCNN format
    print("Exporting trained model to NCNN...")
    model.export(format="ncnn")
    
    # 2. Load the newly exported NCNN model
    print("Loading NCNN model...")
    ncnn_model = YOLO("runs/detect/visdrone_detection/weights/best_ncnn_model")
    
    return ncnn_model

if __name__ == '__main__':
    # Uncomment the function you want to run.
    
    # 1. Train OBB Model
    # train_obb_model()
    
    # 2. Train YOLO-WorldV2 Model
    train_world_model()
    
    # 3. Setup and Run Tracking (NCNN Inference)
    setup_and_run_tracking()

    # 4. Detection Model Training
    detection()
    
    print("Script is ready. Please uncomment the function you wish to execute in the __main__ block.")

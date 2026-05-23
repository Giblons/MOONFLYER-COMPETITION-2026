import os
from ultralytics import YOLO

# --- CONFIGURATION ---
# Define paths to your trained .pt models
DET_MODEL_PATH = "/home/rizky/object-detection/runs/detect/visdrone_detection/weights/best.pt"
CLS_MODEL_PATH = "/home/rizky/object-detection/runs/classify/vcor_visdrone_detection_augmented-2/weights/best.pt"

def export_all():
    """
    Exports both detection and classification models to ONNX, MNN, and NCNN.
    """
    models = {
        "Detection": DET_MODEL_PATH,
        "Classification": CLS_MODEL_PATH
    }
    
    # User requested: ONNX, MNN, NCNN
    formats = ["onnx", "mnn", "ncnn"]
    
    for name, path in models.items():
        if not os.path.exists(path):
            print(f"❌ Error: {name} model not found at {path}")
            continue
            
        print(f"\n--- Processing {name} Model: {path} ---")
        model = YOLO(path)
        
        for fmt in formats:
            print(f"\nExporting {name} to {fmt}...")
            try:
                # Export the model
                # Note: half=True enables FP16, which is standard for edge deployment (MNN/NCNN)
                exported_path = model.export(format=fmt)
                print(f"✅ {name} exported to {fmt} successfully.")
                print(f"   Output: {exported_path}")
            except Exception as e:
                print(f"❌ Failed to export {name} to {fmt}: {e}")

if __name__ == "__main__":
    export_all()
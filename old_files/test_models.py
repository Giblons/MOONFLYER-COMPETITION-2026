import os
from ultralytics import YOLO

def test_four_models():
    """
    Test four different trained models on four different downloaded YouTube videos.
    Make sure to update the 'video_paths' below with the actual paths to your downloaded videos.
    """
    print("Initializing Multi-Model Testing...")

    # Define the paths to your 4 trained models
    model_paths = [
        "/home/giblon/object-detection/runs/detect/visdrone_detection/weights/best.pt",
        "/home/giblon/object-detection/runs/detect/visdrone_world/weights/best.pt",
        #"/home/giblon/object-detection/runs/detect/visdrone_world2/weights/best.pt",
        #"/home/giblon/object-detection/runs/detect/visdrone_world3/weights/best.pt"
    ]

    # Define the paths to your 4 downloaded YouTube videos
    video_paths = [
        "./dataset_video/dataset1.mp4",
        "./dataset_video/dataset2.mp4",
        #"./dataset_video/dataset3.mp4",
        #"./dataset_video/dataset4.mp4"
    ]

    # Ensure we have the same number of models and videos
    assert len(model_paths) == len(video_paths), "You must provide exactly 4 video paths for the 4 models."

    for i in range(4):
        m_path = model_paths[i]
        v_path = video_paths[i]

        print(f"\n{'='*50}")
        print(f"Test #{i+1}")
        print(f"Model: {m_path}")
        print(f"Video: {v_path}")
        print(f"{'='*50}")

        # Basic check to see if the model exists before trying to load it
        if not os.path.exists(m_path):
            print(f"❌ Error: Model file not found at {m_path}")
            print("Skipping to next test...\n")
            continue
            
        # Basic check for video file
        if not os.path.exists(v_path):
            print(f"❌ Error: Video file not found at {v_path}")
            print(f"Please update the 'video_paths' list in this script with the correct path to video #{i+1}.")
            print("Skipping to next test...\n")
            continue

        try:
            # Load the custom-trained model
            model = YOLO(m_path)
            
            # Perform tracking
            print("Starting inference...")
            results = model.track(source=v_path, show=True, device=0)
            
            print(f"✅ Finished testing Model #{i+1} on Video #{i+1}")
            
        except Exception as e:
            print(f"❌ An error occurred during tracking: {e}")

if __name__ == '__main__':
    test_four_models()

import cv2
import threading
import time

class CameraCaptureThread(threading.Thread):
    """
    Capture Thread: Reads a continuous video stream from the MIPI camera
    using cv2.VideoCapture. It places the latest frame into a thread-safe,
    mutex-locked buffer, actively discarding stale frames to ensure zero-latency processing.
    Targeting RADXA Zero 3W (Rockchip RK3566, ARM Cortex-A53).
    """
    def __init__(self, camera_index=0, width=1920, height=1080, fps=30):
        super().__init__()
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps

        # GStreamer pipeline string for MIPI camera on Rockchip
        # This pipeline is a common example for V4L2 on such ARM boards.
        self.pipeline = (
            f"v4l2src device=/dev/video{self.camera_index} ! "
            f"video/x-raw, format=NV12, width={self.width}, height={self.height}, framerate={self.fps}/1 ! "
            "videoconvert ! appsink drop=true max-buffers=1"
        )

        # Open video capture
        # Fallback to standard V4L2 if GStreamer fails or is unavailable
        self.cap = cv2.VideoCapture(self.pipeline, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            print(f"[CameraCapture] GStreamer pipeline failed, falling back to V4L2 index {self.camera_index}")
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        self.lock = threading.Lock()
        self.latest_frame = None
        self.running = False
        self.daemon = True # Thread dies when main thread exits

    def run(self):
        self.running = True
        print("[CameraCapture] Capture thread started.")
        while self.running:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                with self.lock:
                    # Update the latest frame, discarding the old one
                    self.latest_frame = frame
            else:
                # If frame drop or error, sleep slightly to yield CPU
                time.sleep(0.01)

    def get_latest_frame(self):
        """
        Pulls the latest frame from the buffer.
        Returns None if no frame is available.
        """
        with self.lock:
            if self.latest_frame is not None:
                return self.latest_frame.copy()
            return None

    def stop(self):
        """Stops the capture thread and releases resources."""
        self.running = False
        self.join()
        if self.cap.isOpened():
            self.cap.release()
        print("[CameraCapture] Capture thread stopped.")

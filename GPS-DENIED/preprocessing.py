import cv2
import numpy as np

class Preprocessor:
    """
    Optical Calibration & Advanced Preprocessing
    Implements fisheye undistortion and various thresholding/color space conversions
    to prepare the frame for semantic segmentation and feature matching.
    """
    def __init__(self, K=None, D=None, img_shape=(1920, 1080)):
        # Default intrinsic matrix (K) and distortion coefficients (D)
        # In a real scenario, these would be loaded from a calibration file.
        if K is None:
            # Fake camera matrix for a 1080p frame
            focal_length = 800
            self.K = np.array([[focal_length, 0, img_shape[0]/2],
                               [0, focal_length, img_shape[1]/2],
                               [0, 0, 1]], dtype=np.float32)
        else:
            self.K = np.array(K, dtype=np.float32)

        if D is None:
            # Fake fisheye distortion coefficients
            self.D = np.array([[-0.1], [0.01], [-0.001], [0.0001]], dtype=np.float32)
        else:
            self.D = np.array(D, dtype=np.float32)

        self.img_shape = img_shape

        # Estimate new camera matrix to crop undefined edges (balance=0.0 yields pristine orthogonal ROI)
        self.new_K = cv2.fisheye.estimateNewCameraMatrixForUndistort(
            self.K, self.D, self.img_shape, np.eye(3), balance=0.0
        )

        # Precompute undistortion maps for performance
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
            self.K, self.D, np.eye(3), self.new_K, self.img_shape, cv2.CV_16SC2
        )

    def undistort(self, frame):
        """
        Applies fisheye undistortion to correct wide-angle equidistant lens distortion.
        Returns a pristine orthogonal Region of Interest (ROI).
        """
        # remap is generally faster than undistortImage if maps are precomputed
        return cv2.remap(frame, self.map1, self.map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    def apply_otsu_threshold(self, gray_frame):
        """
        Dynamic Binarization & Thresholding:
        Implements Otsu's Global Thresholding for bimodal illumination.
        """
        # Apply Otsu's thresholding
        _, binary = cv2.threshold(gray_frame, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return binary

    def apply_adaptive_gaussian_threshold(self, gray_frame):
        """
        Implements Adaptive Gaussian Thresholding for handling complex shadows.
        """
        # Adaptive thresholding with block size 11 and C 2
        binary = cv2.adaptiveThreshold(
            gray_frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        return binary

    def convert_color_space(self, frame, space="HSV"):
        """
        Color Space Conversion:
        Converts the live feed into various color spaces (e.g., HSV, LAB) to facilitate
        color thresholding and semantic segmentation against vector databases.
        """
        if space.upper() == "HSV":
            return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        elif space.upper() == "LAB":
            return cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        elif space.upper() == "GRAY":
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            return frame # Default return BGR if unknown

    def preprocess_pipeline(self, frame):
        """
        Full preprocessing pipeline example.
        """
        # 1. Undistort
        undistorted = self.undistort(frame)

        # 2. Color space conversions
        gray = self.convert_color_space(undistorted, "GRAY")
        hsv = self.convert_color_space(undistorted, "HSV")

        # 3. Thresholding (example using Otsu)
        binary = self.apply_otsu_threshold(gray)

        return undistorted, gray, hsv, binary

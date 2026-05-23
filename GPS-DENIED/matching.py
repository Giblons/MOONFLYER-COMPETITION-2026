import cv2
import numpy as np

class Matcher:
    """
    Multi-Modal Feature Matching & Confidence Scoring
    Implements ORB with FLANN (LSH) for raster matching.
    Calculates confidence score based on RANSAC inliers.
    """
    def __init__(self, min_confidence=0.7):
        self.min_confidence = min_confidence

        # Initialize ORB
        # Limit max features to keep performance high on ARM Cortex-A53
        self.orb = cv2.ORB_create(nfeatures=2000)

        # Configure FLANN matcher with LSH
        # algorithm=6 is FLANN_INDEX_LSH
        index_params = dict(algorithm=6,
                            table_number=6,
                            key_size=12,
                            multi_probe_level=1)
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)

    def extract_features(self, image):
        """Extracts ORB keypoints and descriptors."""
        # Ensure image is grayscale for ORB
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        kp, des = self.orb.detectAndCompute(gray, None)
        return kp, des

    def match_features(self, kp1, des1, kp2, des2):
        """
        Matches features using FLANN and applies Lowe's Ratio Test.
        Returns good matches.
        """
        if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
            return []

        # knnMatch returns top 2 matches for each descriptor
        try:
            matches = self.flann.knnMatch(des1, des2, k=2)
        except Exception as e:
            # Catch FLANN errors (e.g. empty descriptors)
            return []

        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                # Lowe's Ratio Test (ratio < 0.7)
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        return good_matches

    def calculate_homography_and_confidence(self, kp1, kp2, good_matches):
        """
        Calculates Homography using RANSAC and determines confidence score.
        Confidence is based on the ratio of inliers to total good matches.
        Returns (homography_matrix, confidence_score, mask)
        """
        if len(good_matches) < 4:
            return None, 0.0, None

        # Extract location of good matches
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Find homography using RANSAC
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if M is None or mask is None:
            return None, 0.0, None

        inliers = np.sum(mask)
        total_matches = len(good_matches)

        # Confidence Metric: Ratio of inliers to total matches, scaled.
        # Alternatively, use absolute number of inliers, mapped to 0-1.
        # Here we use a hybrid approach
        ratio_confidence = inliers / total_matches if total_matches > 0 else 0

        # Require at least 20 inliers for a good absolute match
        abs_confidence = min(inliers / 20.0, 1.0)

        # Final confidence is weighted
        confidence = (ratio_confidence * 0.4) + (abs_confidence * 0.6)

        return M, confidence, mask

    def process_match(self, live_img, reference_img):
        """
        Full matching pipeline between a live frame and a reference tile.
        Returns (homography_matrix, confidence)
        """
        kp1, des1 = self.extract_features(live_img)
        kp2, des2 = self.extract_features(reference_img)

        good_matches = self.match_features(kp1, des1, kp2, des2)

        H, confidence, mask = self.calculate_homography_and_confidence(kp1, kp2, good_matches)

        # Enforce strict minimum Confidence Level
        if confidence < self.min_confidence:
            return None, confidence

        return H, confidence

    def semantic_match(self, live_binary, vector_shapes):
        """
        Semantic/Vector Matching stub.
        Compares thresholded shapes against vector database geometries.
        """
        # In a full implementation, this would use shape matching (e.g. cv2.matchShapes)
        # or calculate structural similarity between rendered vectors and binary mask.
        confidence = 0.0
        # Dummy implementation
        return None, confidence

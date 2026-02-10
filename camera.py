import cv2 as cv
import numpy as np
from typing import Tuple

class Camera:
    """
    Wrapper around OpenCV's video capture for handling camera input and 
    basic frame preprocessing.

    Responsibilities:
    - Initialize and manage the camera device
    - Read frames from the camera
    - Apply optional image preprocessing filters
    """

    def __init__(self, camera_id: int = 0):
        """
        Initialize the camera capture device.
        
        Args:
            camera_id (int): Index of the camera device.
                             Default is 0 (usually the primary camera)
        """

        self.cap = cv.VideoCapture(camera_id)
        # Camera properties - set to HD
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
        # Ensure the camera was successfully opened
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera")
        
    def read(self) -> Tuple[bool, np.ndarray]:
        """
        Capture a single frame from the camera.
        
        Returns:
            Tuple[bool, np.ndarray]:
                - success (bool): True if the frame was read successfully.
                - frame (np.ndarray): The captured BGR image.
        """
        return self.cap.read()
    
    # TODO - This filter is an experimental test - very aggressive and remove useful details.
    def depth_like_filter(self, frame: np.ndarray) -> np.ndarray:
        """
        Apply an edge-based filter to emphasize foreground objects.

        This method attempts to simulate a depth-like effect by:
        - Detecting edges
        - Dilating edges
        - Masking out non-edge regions (background)

        Args:
            frame (np.ndarray): Input BGR image.

        Returns:
            np.ndarray: Image with background suppressed.
        """
        # convert to grayscale
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # blur to reduce noise
        frame_blur = cv.GaussianBlur(frame_gray, (9, 9), 0)
        # detect edges
        frame_edges = cv.Canny(frame_blur, 50, 120)
        # thicken edges 
        kernel = np.ones((5, 5), np.uint8)
        edges = cv.dilate(frame_edges, kernel, iterations=2)

        mask = edges > 0
        result = frame.copy()
        # Suppress background
        result[~mask] = (0, 0, 0)

        return result
    
    # TODO - This filter is an experimental test - highly sensitive to lighting conditions.
    def skin_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Generate a skin-color mask using the YCrCb color space.

        This method performs a basic threshold-based skin segmentation.
        It is highly dependent on lighting and camera characteristics
        and should be considered experimental.

        Args:
            frame (np.ndarray): Input BGR image.

        Returns:
            np.ndarray: Binary mask where skin-like regions are white.
        """
        ycrcb = cv.cvtColor(frame, cv.COLOR_BGR2YCrCb)

        lower = np.array([0, 135, 85])
        upper = np.array([255, 180, 135])

        mask = cv.inRange(ycrcb, lower, upper)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        mask = cv.morphologyEx(mask, cv.MORPH_DILATE, kernel)

        return mask

    def release(self):
        """
        Release the camera resource.

        Must be called when the camera is no longer needed.
        """
        self.cap.release()
    
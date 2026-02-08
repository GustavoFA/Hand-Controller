import cv2 as cv
import numpy as np
from typing import Tuple

class Camera:
    """
    Camera control using OpenCV.
    """

    def __init__(self, camera_id:int=0):
        """
        Initialize the camera.
        
        :param camera_id: Index of the camera device
        :type camera_id: int
        """

        self.cap = cv.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera")
        
    def read(self) -> Tuple[bool, np.ndarray]:
        """
        Read a frame from the camera.
        
        :return: Current frame
        :rtype: Tuple[bool, np.ndarray]
        """
        return self.cap.read()
    
    # TODO - not good - too strong
    def depth_like_filter(self, frame) -> np.ndarray:

        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        frame_blur = cv.GaussianBlur(frame_gray, (9, 9), 0)

        frame_edges = cv.Canny(frame_blur, 50, 120)

        kernel = np.ones((5, 5), np.uint8)
        edges = cv.dilate(frame_edges, kernel, iterations=2)

        mask = edges > 0

        result = frame.copy()
        result[~mask] = (0, 0, 0) # destroy background

        return result
    
    # TODO - terrible
    def skin_mask(self, frame) -> np.ndarray:
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
        """
        self.cap.release()
    
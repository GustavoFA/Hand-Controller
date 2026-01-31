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
    
    def release(self):
        """
        Release the camera resource.
        """
        self.cap.release()
    
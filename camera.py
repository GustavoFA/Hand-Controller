import cv2 as cv

class Camera:

    def __init__(self, camera_id:int=0):
        self.cap = cv.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera")
        
    def read(self):
        return self.cap.read()
    
    def release(self):
        self.cap.release()
    
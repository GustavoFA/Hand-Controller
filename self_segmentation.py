import cv2 as cv
import numpy as np
import mediapipe as mp

class SelfSegmentationTools:

    def __init__(self):
        
        self.mp_selfie = mp.solutions.self_segmentation
        self.segmenter = self.mp_selfie.SelfSegmentation(model_selection=1)

    def segment_foreground(self, frame:np.ndarray) -> np.ndarray:
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        result = self.segmenter.process(rgb)

        mask = result.segmentation_mask > 0.6

        output = frame.copy()
        output[~mask] = (0, 0, 0)

        return output
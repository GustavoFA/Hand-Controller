import cv2 as cv
import numpy as np
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from mediapipe import Image, ImageFormat

class SelfSegmentationTools:

    def __init__(self, model:"selfie_segmenter.tflite"):
        
        self.base_options = python.BaseOptions(model_asset_path=model)
        self.options = vision.ImageSegmenterOptions(base_options=self.base_options,
                                       output_category_mask=True)
        self.segmenter = vision.ImageSegmenter.create_from_options(self.options)

    # TODO - not finished
    # Access this links - https://ai.google.dev/edge/mediapipe/solutions/vision/image_segmenter/python#live-stream
    def segment_foreground(self, frame:np.ndarray) -> np.ndarray:
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        result = self.segmenter.process(rgb)

        mask = result.segmentation_mask > 0.6

        output = frame.copy()
        output[~mask] = (0, 0, 0)

        return output
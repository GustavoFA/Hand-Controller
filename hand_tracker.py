import cv2 as cv
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import Image, ImageFormat

class HandTracker:

    MARGIN = 10  # pixels
    FONT_SIZE = 1
    FONT_THICKNESS = 1
    HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

    def __init__(self, model_path:str="hand_landmarker.task", num_hands:int=1):
        # creating the handlandmarker objetct
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(base_options=base_options,
                                            # running_mode=vision.RunningMode.LIVE_STREAM,
                                            #  result_callback=print_ # fix this to better returns
                                            num_hands=num_hands)
        self.detector = vision.HandLandmarker.create_from_options(options) 

        # visualization utilities
        self.mp_hands = mp.tasks.vision.HandLandmarksConnections
        self.mp_drawing = mp.tasks.vision.drawing_utils
        self.mp_drawing_styles = mp.tasks.vision.drawing_styles

    def detect(self, frame):
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        mp_image = Image(
            image_format=ImageFormat.SRGB,
            data=rgb_frame
        )

        result = self.detector.detect(mp_image)

        return result, mp_image
    
    def draw_landmarks_on_image(self, rgb_image, detection_result):
        hand_landmarks_list = detection_result.hand_landmarks
        handedness_list = detection_result.handedness
        annotated_image = np.copy(rgb_image)

        # Loop through the detected hands to visualize.
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            handedness = handedness_list[idx]

            # Draw the hand landmarks.
            self.mp_drawing.draw_landmarks(
            annotated_image,
            hand_landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing_styles.get_default_hand_landmarks_style(),
            self.mp_drawing_styles.get_default_hand_connections_style())

            # Get the top left corner of the detected hand's bounding box.
            height, width, _ = annotated_image.shape
            x_coordinates = [landmark.x for landmark in hand_landmarks]
            y_coordinates = [landmark.y for landmark in hand_landmarks]
            text_x = int(min(x_coordinates) * width)
            text_y = int(min(y_coordinates) * height) - self.MARGIN

            # Draw handedness (left or right hand) on the image.
            cv.putText(annotated_image, f"{handedness[0].category_name}",
                        (text_x, text_y), cv.FONT_HERSHEY_DUPLEX,
                        self.FONT_SIZE, self.HANDEDNESS_TEXT_COLOR, self.FONT_THICKNESS, cv.LINE_AA)

        return annotated_image
    
    def close(self):
        self.detect.close()
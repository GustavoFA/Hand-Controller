import time
import math
import cv2 as cv
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple
from mediapipe.tasks import python
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python import vision

class HandTracker:
    """
    
    """

    MARGIN = 10  # pixels
    FONT_SIZE = 1
    FONT_THICKNESS = 1
    HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

    HAND_KNUCKLES = ['WRIST', 'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP', 'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP',
                    'INDEX_FINGER_TIP', 'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP', 'RING_FINGER_MCP',
                    'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP', 'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP']
    FINGER_INDEX = {
        'thumb': (4, 3),
        'index': (5, 8),
        'middle': (9, 12),
        'ring': (13, 16),
        'pinky': (17, 20)
    }
    HAND_KNUCKLES_COORDINATES = []

    def __init__(self, model_path:str="hand_landmarker.task", 
                 mode:str="live_stream", 
                 num_hands:int=1,
                 min_hand_detection_confidence:float=0.2,
                 min_hand_presence_confidence:float=0.2,
                 min_tracking_confidence:float=0.2
                 ):
        # creating the handlandmarker objetct
        base_options = python.BaseOptions(model_asset_path=model_path)
        # selecting the mode of hand landmark
        self.mode = mode.lower()
        if self.mode == "live_stream":
            options = vision.HandLandmarkerOptions(base_options=base_options,
                                                running_mode=vision.RunningMode.LIVE_STREAM,
                                                num_hands=num_hands,
                                                min_hand_detection_confidence=min_hand_detection_confidence,
                                                min_hand_presence_confidence=min_hand_presence_confidence,
                                                min_tracking_confidence=min_tracking_confidence,
                                                result_callback=self.update_results, 
                                                )
        elif self.mode == "video":
            options = vision.HandLandmarkerOptions(base_options=base_options,
                                                   running_mode=vision.RunningMode.VIDEO,
                                                   num_hands=num_hands,
                                                    min_hand_detection_confidence=min_hand_detection_confidence,
                                                    min_hand_presence_confidence=min_hand_presence_confidence,
                                                    min_tracking_confidence=min_tracking_confidence
                                                    )
        else:
            raise ValueError("Mode not available. The options are: live_stream and video.")

        self.detector = vision.HandLandmarker.create_from_options(options) 

        # visualization utilities
        self.mp_hands = mp.tasks.vision.HandLandmarksConnections
        self.mp_drawing = mp.tasks.vision.drawing_utils
        self.mp_drawing_styles = mp.tasks.vision.drawing_styles

        # handlandmarks result
        self.results = vision.HandLandmarkerResult

        self._timestamp_ms = 0

    @staticmethod
    def _distance_2d(a, b) -> float:
        return math.hypot(a.x - b.x, a.y - b.y)
    
    def update_results(self, result: vision.HandLandmarkerResult, output_image: Image, timestamp: int):
        self.results = result

    def get_results(self) -> vision.HandLandmarkerResult:
        return self.results

    def _detect_async(self, frame):
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

        mp_image = Image(
            image_format=ImageFormat.SRGB,
            data=rgb_frame
        )

        self._timestamp_ms += 1 # guaranteed monotonic

        self.detector.detect_async(image=mp_image,
                                   timestamp_ms=self._timestamp_ms
                                #    timestamp_ms=int(time.time()*1000)
        )

    # TODO - 
    def is_tweezers(self, threshold:float, target_score:float=0.98) -> bool:
        result = self.get_results()

        if not result or not result.hand_landmarks:
            return False
        
        # Take the first hand
        if result.handedness[0][0].score < target_score:
            return False
        
        landmarks = result.hand_landmarks[0]

        # 4 represents thumb tip and 8 represents index finger tip
        distance = self._distance_2d(landmarks[4], landmarks[8])
        print(distance)

        return distance < threshold

    # TODO - this function works with the palm, but back of the hand doesn't work
    def is_finger_extended(self, finger:str) -> bool:
        finger = finger.lower()
        if len(self.HAND_KNUCKLES_COORDINATES ) < 21:
            print("Couldn't find the knuckles coordinates")
            return False
        if finger not in self.FINGER_INDEX.keys():
            print(f"Finger name unavailable - Possible names : {list(self.FINGER_INDEX.keys())}")
            return False
        # for thumb we should use the X axis and for other fingers the Y axis
        if finger == 'thumb':
            axis = 0 # X
        else:
            axis = 1 # Y
        return True if self.HAND_KNUCKLES_COORDINATES[self.FINGER_INDEX[finger][1]][axis] < self.HAND_KNUCKLES_COORDINATES[self.FINGER_INDEX[finger][0]][axis] else False


    # TODO - check if this fuction is useful
    # ABANDONED
    def get_knuckle_coordinates(self, region:str|int, target_score:float=0.98) -> Optional[Tuple[float, float, float]]:
        detection_result = self.get_results()
        try:
            if not detection_result or not detection_result.hand_landmarks:
                print("Couldn't find any hand landmark result")
                return None
            else:
                
                if isinstance(region, int):
                    if 0 <= region <= 20:
                        pass
                    else:
                        print('Region coordinate out of value')
                        return None
                elif isinstance(region, str):
                    try:
                        region = self.HAND_KNUCKLES.index(region.upper())
                    except ValueError as e:
                        print(f'Region name not find - {e}')
                        return None
                else:
                    print("Region not accepeted")
                    return None

                hand_landmarks_list = detection_result.hand_landmarks
                handedness_list = detection_result.handedness

                # get just the first hand detected
                if handedness_list[0][0].score < target_score:
                    print(f"Handedness score too low - {handedness_list[0][0].score}")
                    return None

                return hand_landmarks_list[0][region].x, hand_landmarks_list[0][region].y, hand_landmarks_list[0][region].z

        except AttributeError as e:
            print(f"ERROR - {e}")
            return None

    # TODO:
        # just take the first hand 
    def update_knuckles_coordinates(self, target_score:float=0.98, verbose:bool=True) -> bool:
        detection_result = self.get_results()
        try:
            if not detection_result.hand_landmarks:
                if verbose: print("Couldn't find any hand landmark")
                return False
            else: 
                # take the first hand
                hand_landmarks = detection_result.hand_landmarks[0]
                handedness = detection_result.handedness[0][0].display_name # it's not useful for our case
                hand_score = detection_result.handedness[0][0].score 

                if hand_score < target_score:
                    print(f"Hand {handedness} detected, but with not enough score --> {hand_score} / {target_score}")
                    return False

                self.HAND_KNUCKLES_COORDINATES = [(landmark.x, landmark.y, landmark.z) for landmark in hand_landmarks]
                return True

        except AttributeError as e:
            print(f"ERROR - {e}")
            return False


    # TODO - Fix this one - just print all the value and hand_score should be class parameter
    def print_positions(self, detection_result: vision.HandLandmarkerResult) -> None:
        try:
            if detection_result.hand_landmarks == []:
                return None
            else:
                hand_landmarks_list = detection_result.hand_landmarks
                handedness_list = detection_result.handedness
                print(20*"=")
                for idx in range(len(hand_landmarks_list)):
                    hand_landmarks = hand_landmarks_list[idx]
                    handedness = handedness_list[idx][0].display_name
                    hand_score = handedness_list[idx][0].score
                    if hand_score > 0.97:
                        print(handedness)
                        print(f'{self.HAND_KNUCKLES[0]} - x:{hand_landmarks[0].x} | y:{hand_landmarks[0].y}')
                        print(f'{self.HAND_KNUCKLES[4]} - x:{hand_landmarks[4].x} | y:{hand_landmarks[4].y}')
                        print(f'{self.HAND_KNUCKLES[8]} - x:{hand_landmarks[8].x} | y:{hand_landmarks[8].y}')
                        print(f'{self.HAND_KNUCKLES[12]} - x:{hand_landmarks[12].x} | y:{hand_landmarks[12].y}')
                        print(f'{self.HAND_KNUCKLES[16]} - x:{hand_landmarks[16].x} | y:{hand_landmarks[16].y}')
                        print(f'{self.HAND_KNUCKLES[20]} - x:{hand_landmarks[20].x} | y:{hand_landmarks[20].y}')
                    # print(f'Handedness: {handedness[0].display_name} [{handedness[0].score:.4f}]\n') # take the hands which the score is more than 0.99
                    # index_finger = hand_landmarks[9]
                    # print(f'{index_finger.x}\n{index_finger.y}\n{index_finger.z}')
                    
                    # for i, knuckles in enumerate(hand_landmarks):
                    #     print(10*'-')
                    #     print(f'{self.HAND_KNUCKLES[i]} : {knuckles}')
                    # print(10*'-')
                print(20*"=")
        except:
            return None

    # TODO - Finish comments
    def draw_landmarks_on_image(self, rgb_image):
        '''
        Docstring for draw_landmarks_on_image
        
        :param self: Description
        :param rgb_image: Description
        '''
        detection_result = self.get_results()
        try:
            if detection_result.hand_landmarks == []:
                return rgb_image
            else:
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
        except:
            return rgb_image

    def close(self):
        self.detector.close()
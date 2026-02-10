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
    Hand tracking and gesture analysis using MediaPipe Hand Landmarker.

    Responsibilities:
    - Run MediaPipe hand landmark detection (live stream or video mode)
    - Track hand landmark coordinates
    - Provide gesture-level utilities (finger extended, tweezers gesture)
    - Draw hand landmarks for visualization
    """

    # Drawing configuration
    MARGIN = 10  # pixels
    FONT_SIZE = 1
    FONT_THICKNESS = 1
    HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green
    
    # MediaPipe hand landmark names (index-aligned) 
    HAND_KNUCKLES = ['WRIST', 'THUMB_CMC', 'THUMB_MCP', 'THUMB_IP', 'THUMB_TIP', 'INDEX_FINGER_MCP', 'INDEX_FINGER_PIP', 'INDEX_FINGER_DIP',
                    'INDEX_FINGER_TIP', 'MIDDLE_FINGER_MCP', 'MIDDLE_FINGER_PIP', 'MIDDLE_FINGER_DIP', 'MIDDLE_FINGER_TIP', 'RING_FINGER_MCP',
                    'RING_FINGER_PIP', 'RING_FINGER_DIP', 'RING_FINGER_TIP', 'PINKY_MCP', 'PINKY_PIP', 'PINKY_DIP', 'PINKY_TIP']
    # Landmark indices used to determine finger extension
    # (mcp_index, tip_index)
    FINGER_INDEX = {
        'thumb': (4, 3), # special : (tip_index, ip_index)
        'index': (5, 8),
        'middle': (9, 12),
        'ring': (13, 16),
        'pinky': (17, 20)
    }
    # Store the most recent hand landmark coordinates (normalized)
    HAND_KNUCKLES_COORDINATES = []

    def __init__(self, model_path:str="hand_landmarker.task", 
                 mode:str="live_stream", 
                 num_hands:int=1,
                 min_hand_detection_confidence:float=0.2,
                 min_hand_presence_confidence:float=0.2,
                 min_tracking_confidence:float=0.2
                 ):
        """
        Initialize the MediaPipe Hand Landmarker.

        Args:
            model_path (str): Path to the MediaPipe hand landmarker model (hand_landmarker.task).
            mode (str): 'live_stream' or 'video'.
            num_hands (int): Maximum number of hands to detect.
            min_hand_detection_confidence (float): Detection confidence threshold.
            min_hand_presence_confidence (float): Presence confidence threshold.
            min_tracking_confidence (float): Tracking confidence threshold.
        """
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

        # latest handlandmarks result
        self.results = vision.HandLandmarkerResult
        # monotonically increasing timestamp (required by MediaPipe)
        self._timestamp_ms = 0

    # ------------------------------------------------------------------
    # Utility functions
    # ------------------------------------------------------------------

    @staticmethod
    def _distance_2d(a: float, b: float) -> float:
        """
        Compute Euclidean distance between two 2D landmarks.
        """
        return math.hypot(a.x - b.x, a.y - b.y)
    
    def update_results(self, result: vision.HandLandmarkerResult, output_image: Image, timestamp: int):
        """
        Callback for live stream mode.
        Updates the latest detection result.
        """
        self.results = result

    # ------------------------------------------------------------------
    # Detection interface
    # ------------------------------------------------------------------

    def get_results(self, frame: np.ndarray) -> vision.HandLandmarkerResult:
        """
        Run hand landmark detection on a frame.

        In live_stream mode, detection is asynchronous.
        In video mode, detection is synchronous.

        Args:
            frame (np.ndarray): Input BGR image.

        Returns:
            vision.HandLandmarkerResult: Latest detection result.
        """
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mp_image = Image(
            image_format=ImageFormat.SRGB,
            data=rgb_frame
        )
        self._timestamp_ms += 1
        if self.mode == 'live_stream':
            self.detector.detect_async(
                image=mp_image,
                timestamp_ms=self._timestamp_ms
            )
            return self.results
        else:
            self.results = self.detector.detect_for_video(mp_image, self._timestamp_ms)
            return self.results
        
    # ------------------------------------------------------------------
    # Gesture detection
    # ------------------------------------------------------------------

    # TODO - Find the threshold values
    def is_tweezers(self, threshold: float, target_score: float = 0.98) -> bool:
        """
        Detect a tweezers gesture (thumb tip close to index tip).

        Args:
            threshold (float): Distance threshold for pinch detection.
            target_score (float): Minimum handedness confidence.

        Returns:
            bool: True if tweezers gesture is detected.
        """
        # result = self.get_results()

        if not self.results or not self.results.hand_landmarks:
            return False
        
        # Take the first hand
        if self.results.handedness[0][0].score < target_score:
            return False
        
        landmarks = self.results.hand_landmarks[0]

        # 4 represents thumb tip and 8 represents index finger tip
        distance = self._distance_2d(landmarks[4], landmarks[8])
        print(distance)

        return distance < threshold

    # TODO - this function works with the palm, but back of the hand doesn't work
    def is_finger_extended(self, finger: str) -> bool:
        """
        Determine whether a specific finger is extended.

        Uses relative landmark positions:
        - Thumb: X-axis comparison
        - Other fingers: Y-axis comparison

        Args:
            finger (str): Finger name ('thumb', 'index', 'middle', 'ring', 'pinky').

        Returns:
            bool: True if the finger is extended.
        """
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
        # return True if self.HAND_KNUCKLES_COORDINATES[self.FINGER_INDEX[finger][1]][axis] < self.HAND_KNUCKLES_COORDINATES[self.FINGER_INDEX[finger][0]][axis] else False
        return (
            self.HAND_KNUCKLES_COORDINATES[self.FINGER_INDEX[finger][1]][axis] 
            < self.HAND_KNUCKLES_COORDINATES[self.FINGER_INDEX[finger][0]][axis]
        )

    def update_knuckles_coordinates(self, target_score: float = 0.98, verbose: bool = True) -> bool:
        """
        Update cached landmark coordinates for the first detected hand.

        Args:
            target_score (float): Minimum handedness confidence.
            verbose (bool): Enable logging.

        Returns:
            bool: True if coordinates were updated successfully.
        """
        try:
            if not self.results.hand_landmarks:
                if verbose: print("Couldn't find any hand landmark")
                return False
            else: 
                # take the first hand
                hand_landmarks = self.results.hand_landmarks[0]
                handedness = self.results.handedness[0][0].display_name # it's not useful for our case
                hand_score = self.results.handedness[0][0].score 

                if hand_score < target_score:
                    print(f"Hand {handedness} detected, but with not enough score --> {hand_score} / {target_score}")
                    return False

                self.HAND_KNUCKLES_COORDINATES = [
                    (landmark.x, landmark.y, landmark.z) 
                    for landmark in hand_landmarks
                    ]
                return True

        except AttributeError as e:
            print(f"ERROR - {e}")
            return False


    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    # TODO - Print in a more visible format - graph?
    def print_positions(self, detection_result: vision.HandLandmarkerResult) -> None:
        """
        Print the hand landmarks coordinates.
        """
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


    def draw_landmarks_on_image(self, rgb_image: np.ndarray) -> np.ndarray:
        """
        Draw hand landmarks and handedness label on an image.

        Args:
            rgb_image (np.ndarray): RGB image.

        Returns:
            np.ndarray: Annotated image.
        """

        try:
            if self.results.hand_landmarks == []:
                return rgb_image
            else:
                hand_landmarks_list = self.results.hand_landmarks
                handedness_list = self.results.handedness
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
        """
        Release MediaPipe resources.
        """
        self.detector.close()
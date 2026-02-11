import time
import cv2 as cv
import pyautogui
from camera import Camera
from hand_tracker import HandTracker
from controller import ComputerInputController
from self_segmentation import SelfSegmentationTools

class HandControlApp:
    """
    A collection of control forms:
    - joystick
    - keyboard
    - hand mouse
    """

    def __init__(self):
        self.camera = Camera()
        self.detector = HandTracker(mode='video', num_hands=1,
                                    min_hand_detection_confidence=0.5, # lower precision -> faster tracking
                                    min_hand_presence_confidence=0.5,
                                    min_tracking_confidence=0.5)
        self.controller = ComputerInputController()
        # self.segmenter_tool = SelfSegmentationTools() # to be explored

    def run_controller_for_game(self, minimum_hand_score:float=0.5, skip_frame:bool=True):
        commands = {
            'index': 'space',
            'thumb': 'd',
            'pinky': 'a',
            'middle': 'w'
        }
        key_states = {finger: False for finger in commands}
        input('Press ENTER to start the controller:\n')
        frame_count = 0
        while True:
            ret, frame = self.camera.read()
            frame_count += 1
            if not ret:
                print('Failed to read frame')
                break
            self.detector.get_results(frame)

            if not self.detector.update_knuckles_coordinates(minimum_hand_score, verbose=False) or (skip_frame and frame_count % 2 != 0):
                continue

            for finger, key in commands.items():
                extended = self.detector.is_finger_extended(finger)
                # PRESS once
                if extended and not key_states[finger]:
                    pyautogui.keyDown(key)
                    key_states[finger] = True
                # RELEASE once
                elif not extended and key_states[finger]:
                    pyautogui.keyUp(key)
                    key_states[finger] = False
        self.cleanup()

    def run_computer_interface(self, minimum_hand_score:float=0.3, skip_frame:bool=True):

        while True:
            ret, frame = self.camera.read()
            if not ret:
                print("Failed to read frame")
                break
            self.detector.get_results(frame)

            cv.imshow("Camera", frame)
            if cv.waitKey(1) != -1:
                break

            if not self.detector.update_knuckles_coordinates(minimum_hand_score, verbose=False):
                continue    
            
            if self.detector.is_finger_extended('index'):
                x, y, _ = self.detector.HAND_KNUCKLES_COORDINATES[8]
                # self.controller.straight_move(x, y)
                self.controller.smooth_move(x, y)

        self.cleanup()


    def cleanup(self):
        self.camera.release()
        self.detector.close()
        cv.destroyAllWindows()

if __name__ == "__main__":

    app = HandControlApp()
    app.run_computer_interface()
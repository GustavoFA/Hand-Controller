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
    - joystick (run_controller_for_game)
    - mouse (run_computer_interface)
    - keyboard (run_keyboard)
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
        """
        Hand-based game controller.

        This method captures frames from the camera, detects hand landmarks,
        and maps specific finger gestures to keyboard key presses.

        Args:
            minimum_hand_score (float):
                Minimum confidence score required to consider the hand detection valid.
            skip_frame (bool):
                If True, processes every other frame to reduce latency and CPU usage.
        """
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

    def run_computer_interface(self, minimum_hand_score:float=0.3):
        """
        Hand-gesture-based computer interface.

        This method captures frames from the camera, detects hand landmarks,
        and maps specific gestures to mouse actions:

            - Index finger → Move cursor
            - Pincer grasp (thumb + index) → Left mouse click (hold)
            - Pinky finger → Right mouse click
            - Index + Middle fingers → Scroll

        The loop runs continuously until the user presses the space key.

        Args:
            minimum_hand_score (float):
                Minimum confidence score required to accept hand detection.
                Frames below this threshold are ignored.
        """
        lmb_pressed = False
        rmb_pressed = False
        index_beta = 0.22
        while True:
            ret, frame = self.camera.read()
            if not ret:
                print("Failed to read frame")
                break
            self.detector.get_results(frame)

            cv.imshow("Camera", frame)
            # Stop with space key
            if cv.waitKey(1) == 32: 
                break

            if not self.detector.update_knuckles_coordinates(minimum_hand_score, verbose=False):
                continue    
            
            if self.detector.is_two_finger_extended(['index', 'middle'], 0.2):
                x_scroll, y_scroll, _ = self.detector.HAND_KNUCKLES_COORDINATES[12]
                self.controller.scroll(x_scroll, y_scroll)
                continue
            elif self.detector.is_finger_extended('index', index_beta): # It doesn't work when pincer grasp
                x, y, _ = self.detector.HAND_KNUCKLES_COORDINATES[8]
                self.controller.smooth_move(x, y)

            # LMB click
            pinched = self.detector.is_tweezers(0.04)
            if pinched and not lmb_pressed:
                pyautogui.mouseDown(button='left')
                lmb_pressed = True
            elif not pinched and lmb_pressed:
                pyautogui.mouseUp(button='left')
                lmb_pressed = False
            
            # RMB click
            rmb_status = self.detector.is_finger_extended('pinky', 0.1)
            if  rmb_status and not rmb_pressed:
                pyautogui.rightClick()
                rmb_pressed = True
            elif not rmb_status and rmb_pressed:
                rmb_pressed = False

        self.cleanup()

    # TODO - Next method (NOT STARTED)
    def run_keyboard(self) -> None:
        """
        Hand-gesture-based virtual keyboard.

        This method will capture hand gestures (e.g., sign language or
        predefined finger configurations) and convert them into
        keyboard text input.
        """
        pass

    def run_debugging(self) -> None:
        """
        Debugging mode
        """
        
        while True:
            ret, frame = self.camera.read()
            if not ret:
                print("Failed to read frame")
                break
            self.detector.get_results(frame)

            cv.imshow("Camera", frame)
            if cv.waitKey(1) != -1:
                break
            
            if not self.detector.update_knuckles_coordinates(0.3, False):
                continue
            
            # if self.detector.is_two_finger_extended(['index', 'middle'], 0.1):
            #     print('two fingers')
            # else:
            #     print('-----')
            # x, y, _ = self.detector.HAND_KNUCKLES_COORDINATES[8]
            # print(
            #     f"INDEX: ({x=}, {y=})"
            #     )
            x, y, _ = self.detector.HAND_KNUCKLES_COORDINATES[12]
            print(
                f"MIDDLE: ({x=}, {y=})"
            )
        

    def cleanup(self):
        self.camera.release()
        self.detector.close()
        cv.destroyAllWindows()

if __name__ == "__main__":

    app = HandControlApp()
    # app.run_debugging()
    # app.run_controller_for_game()
    app.run_computer_interface()
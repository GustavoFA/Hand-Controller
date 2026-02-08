import time
import cv2 as cv
import pyautogui
from camera import Camera
from hand_tracker import HandTracker
from controller import ComputerInputController
from self_segmentation import SelfSegmentationTools

class HandControlApp:

    def __init__(self):
        self.camera = Camera()
        self.detector = HandTracker(num_hands=2)
        self.controller = ComputerInputController()
        # self.segmenter_tool = SelfSegmentationTools()

    def run_controller_for_game(self, minimum_hand_score:float=0.5, debounce:float=0.3):
        commands = {
            'index': 'space',
            'thumb': 'd',
            'pinky': 'a',
            'middle': 'w'
        }

        input('Press ENTER to start the controller:\n')
        last_time = time.time()
        while True:
            ret, frame = self.camera.read()
            if not ret:
                print('Failed to read frame')
                break
            
            self.detector._detect_async(frame)
            if self.detector.update_knuckles_coordinates(minimum_hand_score, verbose=False) and (time.time() - last_time > debounce):
                keys_status = {}
                for finger, button in commands.items():
                    if self.detector.is_finger_extended(finger):
                        keys_status[button] = True
                    else:
                        keys_status[button] = False
                self.controller.controller_buttons(keys_status)
                last_time = time.time()
        self.cleanup()

    def run_test(self):

        run_time = 0
        while True:
            ret, frame = self.camera.read()
            if not ret:
                print("Failed to read frame")
                break
            
            # frame = self.camera.depth_like_filter(frame)
            # frame = self.camera.skin_mask(frame)
            # frame = self.segmenter_tool.segment_foreground(frame)

            # update the hand position
            self.detector.detect_async(frame)

            # results = self.detector.get_results()

            # try:
            #     x, y, _ = self.detector.get_knuckle_coordinates(8)
            #     # self.controller.smooth_move(x, y)
            #     self.controller.straight_move(x, y)
            # except TypeError as e:
            #     print(e)

            if self.detector.update_knuckles_coordinates(0.5) and (time.time() - run_time > 1.0):
                run_time = time.time()

                print(self.detector.HAND_KNUCKLES_COORDINATES[0])

                # INDEX_F = self.detector.HAND_KNUCKLES_COORDINATES[5:9]
                # # print("\nCOORDINATES:")
                # # # print(INDEX_F)

                # THUMB_F = self.detector.HAND_KNUCKLES_COORDINATES[1:5]
                # print('\n-----')
                # for pos in THUMB_F:
                #     print(pos)

                # To identify 
                print(10*'-')
                # for key in self.detector.FINGER_INDEX.keys():
                #     if self.detector.is_finger_extended(key):
                #         print(key)
            
            # print(f'\n\n{results}\n\n{len(results)}\n\n')

            # final_frame = self.detector.draw_landmarks_on_image(frame)

            # self.detector.print_positions(results)
            # print(self.detector.get_all_knuckle_coordinates(results))

            # detection_result, mp_image = self.detector.detect(frame)

            # if detection_result.hand_landmarks:
            #     print(f"Hands detected: {len(detection_result.hand_landmarks)}")

            # frame_with_draw = self.detector.draw_landmarks_on_image(
            #     mp_image.numpy_view(),
            #     detection_result
            # )
# 
            # final_frame = cv.cvtColor(frame_with_draw, cv.COLOR_RGB2BGR)
            

            cv.imshow("Camera", frame)

            if cv.waitKey(1) != -1:
                break
        
        self.cleanup()


    def cleanup(self):
        self.camera.release()
        self.detector.close()
        cv.destroyAllWindows()
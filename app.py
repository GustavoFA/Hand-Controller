import cv2 as cv
from camera import Camera
from hand_tracker import HandTracker

class HandControlApp:

    def __init__(self):
        self.camera = Camera()
        self.detector = HandTracker(num_hands=2)

    def run(self):
        
        while True:
            ret, frame = self.camera.read()
            if not ret:
                print("Failed to read frame")
                break

            detection_result, mp_image = self.detector.detect(frame)

            if detection_result.hand_landmarks:
                print(f"Hands detected: {len(detection_result.hand_landmarks)}")

            frame_with_draw = self.detector.draw_landmarks_on_image(
                mp_image.numpy_view(),
                detection_result
            )

            final_frame = cv.cvtColor(frame_with_draw, cv.COLOR_RGB2BGR)

            cv.imshow("Camera", final_frame)

            if cv.waitKey(1) != -1:
                break
        
        self.cleanup()


    def cleanup(self):
        self.camera.release()
        self.detector.close()
        cv.destroyAllWindows()
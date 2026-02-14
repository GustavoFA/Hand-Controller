# Hand Controller

A computer vision project that uses hand gestures to control the mouse and keyboard in real time.

This project leverages MediaPipe Hand Landmarker to detect and track hand landmarks from a camera feed, mapping specific hand poses to system input commands (mouse movement and clicks).

## Features
- Real-time hand landmark detection using MediaPipe
- Gesture-based mouse control
- Configurable gesture → command mapping
- Modular design (camera, hand tracking, input controller)

## Gesture Commands

| Gesture                          | Controller action                  | Mouse action                  |
| -------------------------------- | ---------------------------------- | ----------------------------- |
| Index finger extended            | Space                              | Mouse cursor                  | 
| Middle finger extended           | W                                  | -                             |
| Pinky finger extended            | A                                  | RMB click                     |
| Thumb finger extended            | D                                  | -                             |
| Closed hand                      | Neutral state (no action)          | Neutral state (no action)     |
| Pinching                         | -                                  | LMB click                     |
| Index and middle finger extended | -                                  | Scroll                        | 

## Requirements

- Python version >= 3.10 (Tested with 3.12.3)
- MediaPipe – hand landmark detection
- OpenCV – camera input & image processing
- PyAutoGUI – mouse and keyboard control
- NumPy #conflict - Last time I use 2.2.4

## Model File
You need the MediaPipe Hand Landmarker model. Download it in the link below and place the file in your project root (or update the path in the code).

https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

## Getting Started

```
git clone https://github.com/GustavoFA/Hand-Controller.git
cd Hand-Controller
python -m venv .venv
```
For Linux/MacOS
```
source .venv/bin/activate  
```
For Windows
```
.venv\Scripts\activate
```
Then
```
pip install -r requirements.txt
python main.py
```
Make sure your webcam is connected and accessible.

## Project Status

### Future Improvements

- Support multiple hands
- Use the computer without a mouse and keyboard :
    For this case we could create 4 commands, which are mouse control (index finger extended) [DONE], LMB click (pincer grasp) [DONE], RMB click (pinky externded) [DONE] and scroll wheel (movement of two fingers, index and middle together) [DONE]. 
- Add a "menu" to select the app mode (gamer, mouse or keyboard).
- Allows to define custom commands in gamer mode.
- Implement a keyboard using sign language detection.
- Increase the number of possible commands (currently, there are only 5 - one per finger).

### Issues fixed

- Highly latency : The keyboard commands have a lot of latency between each comand. The main bottleneck appears to be OS-level input calls. This was fixed by reducing OS calls and disable unnecessary PyAutoGUI delays. Another way to reduce latency is by using other libraries, like `pynput`, `evdev` (Linux) or `virtual gamepad` (vgamepad).

- Mouse cursor jittering : When trying to control the mouse cursor with index finger, the cursor was jittering. The solution was to use a smoothing movement with `alpha = 0.2` and fix de X and Y values in the original function. Furthermore, I inverted the X values and fixed the the issue where the screen size was being mixed with mouse cursor coordinates. Another improvement was adding a linear threshold value to the finger-extended detector function, which better separetes when the finger is extended or not.

- Ubuntu doesn't accept virtual input commands - Check if the system is running in Wayland mode (use `echo $XDG_SESSION_TYPE` in the terminal). If so, switch to x11.

- Hand threshold/space issue : To move the mouse, the camera must see the entire hand. Near the edges of the frame, parts of the hand may be cut off, causing detection failure and, consequently, loss of mouse control. This was fixed by introducing a virtual bounding box in camera frame. With this approach, the mouse cursor reaches the screen edges before hand detection fails.

### Issues

- Can't move mouse cursor while pinching.
- LMB misclicks - check the thumb coordinates.
- Loss of finger extension detection - sometimes, when moving the hand with index finger extended, the system temporality fails to detect it and then recovers. This may be related to the `beta` parameter.

## References
- [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide)
- [Hand landmark detection](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
- [Segmenter](https://ai.google.dev/edge/mediapipe/solutions/vision/image_segmenter)


## License
This project is licensed under the MIT License — see the `LICENSE` file for details.
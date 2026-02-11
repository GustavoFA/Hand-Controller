# Hand Controller

A computer vision project that uses hand gestures to control the mouse and keyboard in real time.

This project leverages MediaPipe Hand Landmarker to detect and track hand landmarks from a camera feed, mapping specific hand poses to system input commands (mouse movement and clicks).

## Features
- Real-time hand landmark detection using MediaPipe
- Gesture-based mouse control
- Configurable gesture → command mapping
- Modular design (camera, hand tracking, input controller)

## Gesture Commands

| Gesture                          | Controller action                  |
| -------------------------------- | ---------------------------------- |
| Index finger extended            | Space                              |
| Middle finger extended           | W                                  |
| Pinky finger extended            | A                                  |
| Thumb finger extended            | D                                  |
| Closed hand                      | Neutral state (no action)          |

## Requirements

- Python version >= 3.10 (Tested with 3.12.3)
- MediaPipe – hand landmark detection
- OpenCV – camera input & image processing
- PyAutoGUI – mouse and keyboard control
- NumPy

## Model File
You need the MediaPipe Hand Landmarker model. Download it in the link below and place the file in your project root (or update the path in the code).

https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

## Getting Started

```
git clone https://github.com/GustavoFA/Hand-Controller.git
cd Hand-Controller
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
python main.py
```
Make sure your webcam is connected and accessible.

## Project Status

### Future Improvements

- Add gesture smoothing and debouncing
- Support multiple hands
- Use the computer without a mouse and keyboard :
    For this case we could create 4 commands, which are mouse control (index finger extended), LMB click (pincer grasp), RMB click (pinky externded) and scroll wheel (movement of two fingers, index and middle together). Furthermore, we could add a "menu" to select the app mode.

### Issues fixed

- Highly latency : The keyboard commands have a lot of latency between each comand. The main bottleneck appears to be OS-level input calls. This was fixed by reducing OS calls and disable unnecessary PyAutoGUI delays. Another way to reduce latency is by using other libraries, like `pynput`, `evdev` (Linux) or `virtual gamepad` (vgamepad).

### Issues

- Hand mouse movement is jittery - we solution with smooth moviment (alpha = 0.2) and inverting the X values.

## References
- [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide)
- [Hand landmark detection](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
- [Segmenter](https://ai.google.dev/edge/mediapipe/solutions/vision/image_segmenter)


## License
This project is licensed under the MIT License — see the `LICENSE` file for details.
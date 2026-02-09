# Hand Controller

Projects focusing on computer vision using hands to command computer.

Tool to identify and track hand movement and positions : [MediaPipe](https://chuoling.github.io/mediapipe/)


You can find the hand landmarker file on this [link](https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task)



---

Hand command positions:

* Finger pointing - move the mouse.
* Tweezers by hand - holding left mouse button (LMB).
* Pinky pointing - right mouse button (RMB) click.
* Loser hand - left mouse button (LMB) click.
* Closed hand - neutral position (do nothing).

---

To get more information about:

* [Hand landmark detection](https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
* [Segmenter](https://ai.google.dev/edge/mediapipe/solutions/vision/image_segmenter)

### Project status

The hand detection and the commands are working, but there's a lot of latency between each command. I don't know where the core of the problem is. After changing the hand detection mode, we saw that the issue is not related to hand capture. This was fixed by reducing OS calls and enabling some features. Another way to reduce latency is by using `pynput`, `evdev` (Linux) or `virtual gamepad` (vgamepad).
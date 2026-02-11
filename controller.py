import pyautogui
import numpy as np

class ComputerInputController:
    """
    High-level controler responsible for simulating keyboard and mouse input.

    This class wraps pyautogui functionality and provides:
    - Low-latency keyboard press/release handling
    - Absolute mouse movement
    - Smoothed mouse movement using an EMA

    Typical use case:
        - Map hand gestures or tracking finger coordinates to OS-level inputs.
    """

    def __init__(self, alpha:float=0.2):
        """
        Init the input controller.

        Args:
            alpha (float): Smoothing factor for mouse movement.
                           Range: (0, 1]
                           Lower values = smoother but slower response.
                            alpha=0.1  # smooth
                            alpha=0.3  # balanced
                            alpha=0.6  # fast
        
        """
        # Disable PyAutoGUI failsafe to prevent exceptions when cursor hits screen corners
        pyautogui.FAILSAFE = False
        # Remove artificial delay between PyAutoGUI comands (reducy latency)
        pyautogui.PAUSE = 0 
        # Screen resolution
        self.screen_w, self.screen_h = pyautogui.size()
        # Initial mouse position
        self.prev_x, self.prev_y = pyautogui.position()
        self.prev_x /= self.screen_w
        self.prev_y /= self.screen_h
        # EMA factor
        self.alpha = alpha
        # Scrolling cursor position
        self.scroll_x = None
        self.scroll_y = None
    
    def update_previous_coordinates(self, x: float = None, y: float = None) -> None:
        """
        Update 
        """
        self.scroll_x = x
        self.scroll_y = y

    @staticmethod
    def controller_buttons(commands:dict[str, bool]) -> None:
        """
        Press or release keyboard keys based on a command dictionary.

        Args:
            commands (dict):
                Dictionary mapping keyboard keys to boolean states.
                Example:
                    {
                        'w': True,
                        'a': False,
                        'space': True
                    }
        """
        for key, status in commands.items():
            if status:
                pyautogui.keyDown(key)
            else:
                pyautogui.keyUp(key)

    def scroll(self, x: float, y: float, gama : float = 1.0) -> None:
        """
        Scrolling 
        """
        if self.scroll_x is None:
            self.scroll_x, self.scroll_y = x, y
            return
        if np.abs(self.scroll_x - x) > np.abs(self.scroll_y - y):
            pyautogui.keyDown('shift')
            pyautogui.scroll(gama * (self.scroll_x - x))
            pyautogui.keyUp('shift')
        else:
            pyautogui.scroll(gama * (self.scroll_y - y))

    def straight_move(self, x: float, y: float) -> None:
        """
        Move the mouse cursor directly to a screen position.

        This method performs a fast, unsmoothed movement.

        Args:
            x (float): Normalized horizontal coordinate (0.0 - 1.0)
            y (float): Normalized vertical coordinate (0.0 - 1.0)
        """
        x = int((1 - x) * self.screen_w)
        y = int(y * self.screen_h)
        pyautogui.moveTo(x, y)

    def smooth_move(self, x: float, y: float) -> None:
        """
        Move the mouse cursor using an exponential moving average (EMA)

        This reduces jitter from noisy inputs sources (e.g., hand tracking),
        producing smoother cursor movement at the cost fo slight latency.

        Args:
            x (float): Normalized horizontal coordinate (0.0 - 1.0)
            y (float): Normalized vertical coordinate (0.0 - 1.0)
        """
        if self.prev_x is None:
            self.prev_x, self.prev_y = x, y

        # Apply EMA smoothing
        x = self.alpha * x + (1 - self.alpha) * self.prev_x
        y = self.alpha * y + (1 - self.alpha) * self.prev_y
        self.prev_x, self.prev_y = x, y

        screen_w = int((1 - x) * self.screen_w)
        screen_h = int(y * self.screen_h)

        pyautogui.moveTo(screen_w, screen_h)
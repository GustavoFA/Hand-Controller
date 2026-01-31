import pyautogui

class ComputerInputController:

    def __init__(self, alpha:float=0.2):

        pyautogui.FAILSAFE = False
        
        self.screen_w, self.screen_h = pyautogui.size()
        self.prev_x, self.prev_y = pyautogui.position()
        self.alpha = alpha

    def straight_move(self, x, y):
        x = int(x * self.screen_w)
        y = int(y * self.screen_h)
        pyautogui.moveTo(x, y)

    def smooth_move(self, x, y):
        if self.prev_x is None:
            self.prev_x, self.prev_y = x, y

        x = int(self.alpha * x + (1 - self.alpha) * self.prev_x)
        y = int(self.alpha * y + (1 - self.alpha) * self.prev_y)

        self.prev_x, self.prev_y = x, y
        pyautogui.moveTo(x, y)
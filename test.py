import win32api
import win32con
import time
time.sleep(1)
screen_width = 2560
win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, (screen_width // 2), 0, 0, 0)
time.sleep(2)
win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, -(screen_width // 2), 0, 0, 0)
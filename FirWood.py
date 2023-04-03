import math


import mss.tools
import time
import os
import sys

import numpy as np
import win32api
import win32con
import random
from pynput.mouse import Controller
from pynput.keyboard import Listener, KeyCode
from ultralytics import YOLO
from PIL import Image, ImageDraw

# define Genshin Impact width and height.
# modify this on your own computer.
# after test, I found the center height need to move down.
# decrease 1/5
SCREEN_WIDTH = 2560
SCREEN_HEIGHT = 1440
screen_width = SCREEN_WIDTH
screen_height = SCREEN_HEIGHT
# define the save path of screenshots.
SCREENSHOT_PATH = "2D"
SCREENSHOT_1D_PATH = "1D"
SCREENSHOT_ANO_PATH = "ANO"

screenshots = os.listdir(SCREENSHOT_PATH)
screenshots1D = os.listdir(SCREENSHOT_1D_PATH)
# path of trained model.
MODEL_PATH = "runs/detect/train2/weights/best.pt"
CUT_MODEL_PATH = "runs/detect/Cutable_trees_with_SEAttention/weights/best.pt"

# if screenshot path not exist,create one.
if not os.path.exists(SCREENSHOT_PATH):
    os.makedirs(SCREENSHOT_PATH)

if not os.path.exists(SCREENSHOT_1D_PATH):
    os.makedirs(SCREENSHOT_1D_PATH)

# define Trees(four corner of tree box and the confidence of trees in model)
class Tree:

    def __init__(self, xmin, ymin, xmax, ymax, confidence):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.confidence = confidence
        self.width = self.xmax - self.xmin
        self.sigmoid_width = 0.4 + 0.4 / (1 + np.exp(-(self.width - 65) / 10))

# calculate center of detect box of trees in 2D(x and y)
    # 在实际测试中发现y中心总是太高，而导致欧氏距离受y影响太大，因为树都是竖着长的。
    # 所以对y中心进行一个向下移动。
    def get_2Dcenter(self):
        x_center = int((self.xmin + (self.xmax - self.xmin) / 2) * 1)
        y_center = int((self.ymin + (self.ymax - self.ymin) / 2) * 1)
        return x_center, y_center

    # calculate 2D distance to center of detect box(Euclidean distance)
    def get_2Ddistance_to(self, x, y):
        sigmoid_width = 0.8 + (1.2 - 0.8) / (1 + np.exp(-(self.width - 65) / 10))
        weighted_distance = (math.sqrt((self.get_2Dcenter()[0] - x) ** 2 + (self.get_2Dcenter()[1] - y) ** 2)) // sigmoid_width
        return weighted_distance


# # calculate 1D distance to center of detect box(only in x)
#     def get_1Ddistance_to(self, x):
#         return -(self.get_2Dcenter()[0] - x)
#
# class Tree1D:
#     def __init__(self, position):
#         self.position = position
#
#     def get_1Ddistance_to(self, x):
#         return abs-(self.position - x)

# define way of getting screenshots
def take_screenshot(sct, screenshot_path):
    try:
        img = sct.grab({"top": 0, "left": 0, "width": SCREEN_WIDTH, "height": SCREEN_HEIGHT})
        mss.tools.to_png(img.rgb, img.size, output=screenshot_path)
        return img, screenshot_path

    except Exception as e:
        print(f"Error taking screenshot: {e}")

def take_1Dscreenshot(sct, screenshot_1D_path):
    try:
        img = sct.grab({"top": 0, "left": 0, "width": SCREEN_WIDTH, "height": SCREEN_HEIGHT})
        mss.tools.to_png(img.rgb, img.size, output=screenshot_1D_path)

        return img, screenshot_1D_path

    except Exception as e:
        print(f"Error taking screenshot: {e}")

# change to float
# model = model_dict['model'].float().fuse().eval()

# Move model to cpu or gpu
# line 122 is gpu and line 123 is cpu
# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

device = 'cpu'

try:
    # modify map_location to see cpu or gpu work better
    model = YOLO(MODEL_PATH)
    print("Model load successfully,press t to start and press f to quit")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

print("Loading model,please wait")
try:
    # modify map_location to see cpu or gpu work better
    model_Cut = YOLO(CUT_MODEL_PATH)
    print("Second Model load successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()
# Initialize the mouse
mouse = Controller()


# Initialize the listener of keyboard
"""exit_key = KeyCode(char='f')

start_key = KeyCode(char='t')"""
exit_key = KeyCode.from_char('f')
start_key = KeyCode.from_char('t')
pressed_keys = set()

# define event of keyboard after press

def on_press(key):
    if isinstance(key, KeyCode) and key.char in {'t', 'f'}:
        if key == exit_key:
            print('Exiting...')
            return False
        if key == start_key:
            print('Starting detection...')
            pressed_keys.add(key)
        else:
            pressed_keys.add(key)

# define event of keyboard after release
def on_release(key):
    if key in pressed_keys:
        pressed_keys.remove(key)

listener = Listener(on_press=on_press, on_release=on_release)
listener.start()

moves = {
    "向左前方移动": {"x_offset": int(screen_width // 4), "time_interval": 1.5},
    "向右前方移动": {"x_offset": -int(screen_width // 4), "time_interval": 1.5},
    "左前偏中": {"x_offset": int(screen_width // 8), "time_interval": 1.5},
    "右前偏中": {"x_offset": -int(screen_width // 8), "time_interval": 1.5},
    "左前偏下": {"x_offset": int(screen_width // 8) + int(screen_width // 4), "time_interval": 1.5},
    "右前偏下": {"x_offset": -int(screen_width // 8) - int(screen_width // 4), "time_interval": 1.5},
    "向左移动": {"x_offset": int(screen_width // 2), "time_interval": 1.5},
    "向右移动": {"x_offset": -int(screen_width // 2), "time_interval": 1.5},

}


# Main loop of screenshot
while True:

    try:
        # 检查是否按下退出键
        if exit_key in pressed_keys:
            print('Exiting...')
            sys.exit(0)
        # 检查是否按下开始键
        if start_key in pressed_keys:
            print('Starting detection...')

        else:
            continue  # 如果没有按下开始键，则跳过本次循环

        # 截图
        try:
            with mss.mss() as sct:
                screenshot_path = f"{SCREENSHOT_PATH}/screenshot_{int(time.time())}.png"
                processed_img, screenshot_path = take_screenshot(sct, screenshot_path)
                print(f"Screenshot saved to {screenshot_path}")
                try:
                    with Image.open(screenshot_path) as img:
                        print("Screenshot saved successfully.")
                except Exception as e:
                    print(f"Error saving screenshot: {e}")
        except Exception as e:
            print(f"Error taking screenshot: {e}")
        # 进行模型推理
        try:
            # print(f"Loading image from {screenshot_path}")
            results = model(screenshot_path, **{'device':device})[0]
            # print(f"Model output tensor shape: {results}")
        except Exception as e:
            print(f"Error: {e}")

        # 获取树木列表（类别编号是classitem,0是树1是旅行者）
        Trees = []
        for xyxy, classitem, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
            if int(classitem) == 0:
                # print(f"Class: {classitem}, Confidence: {conf}")
                Trees.append(Tree(int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), float(conf)))

        # Open the screenshot image
        with Image.open(screenshot_path) as img:
            # Create a draw object
            draw = ImageDraw.Draw(img)

            # Loop through the detected trees and draw a box around them
            for tree in Trees:
                draw.rectangle((tree.xmin, tree.ymin, tree.xmax, tree.ymax), outline="red")

            # Save the annotated image
            annotated_path = f"{SCREENSHOT_ANO_PATH}/annotated_{int(time.time())}.png"
            img.save(annotated_path)
            # print(f"Annotated screenshot saved to {annotated_path}")

        # 使用近距离推理
        try:
            # print(f"Loading image from {screenshot_path}")
            results_Cut = model_Cut(screenshot_path, **{'device': device})[0]
            # print(f"Model output tensor shape: {results_Cut}")
        except Exception as e:
            print(f"Error: {e}")

        Cut_x = []
        for xyxy, classitem, conf in zip(results_Cut.boxes.xyxy, results_Cut.boxes.cls, results_Cut.boxes.conf):
            if int(classitem) == 0 and float(conf) > 0.75 and (int(xyxy[2]) - int(xyxy[0])) > 100 and int(xyxy[3]) - int(xyxy[1]) > 450:
                Cut_x.append((int(xyxy[2]) - int(xyxy[0])))

        print(f"树的宽度: {Cut_x} 置信度: {float(conf)} 树的高度: {int(xyxy[3]) - int(xyxy[1])}")
        # 找到距离最近的树木

        if len(Cut_x) > 0:
            print("Tree is close enough, now cutting")
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            time.sleep(2)
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            time.sleep(2)
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
            win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
            time.sleep(2)
            del Cut_x[:]

            # 随机选择一个移动方式
            move_name, move_params = random.choice(list(moves.items()))

            # 根据选择的移动方式移动视角
            win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, move_params["x_offset"], 0, 0, 0)

            # 移动角色并等待一段时间
            win32api.keybd_event(0x53, 0, 0, 0)
            time.sleep(0.5)
            win32api.keybd_event(0x53, 0, win32con.KEYEVENTF_KEYUP, 0)
            win32api.keybd_event(0x57, 0, 0, 0)
            time.sleep(move_params["time_interval"])
            win32api.keybd_event(0x57, 0, win32con.KEYEVENTF_KEYUP, 0)
            time.sleep(move_params["time_interval"])

            # 截图(回到主循环）
            time.sleep(0.5)
            win32api.keybd_event(0x54, 0, 0, 0)
            time.sleep(0.1)
            win32api.keybd_event(0x54, 0, win32con.KEYEVENTF_KEYUP, 0)

        if len(Trees) > 0 and len(Cut_x) == 0:
            # 获取最近的树
            closest_tree = min(Trees, key=lambda e: e.get_2Ddistance_to(*mouse.position))
            # second_close_tree = sorted(Trees, key=lambda e: e.get_2Ddistance_to(*mouse.position))[1]

            # 获取目标位置坐标
            target_X, target_Y = closest_tree.get_2Dcenter()
            target_x, target_y = target_X, target_Y
            current_x, current_y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2

            # 计算距离
            distance2D_to_tree = closest_tree.get_2Ddistance_to(*mouse.position)
            print(f"第一次判断的条件为2D距离小于1000: {distance2D_to_tree}")

            # distance2D_to_2ndtree = second_close_tree.get_2Ddistance_to(*mouse.position)

            # 如果树木距离鼠标的EU坐标小于1000则自动进行瞄准
            if distance2D_to_tree < 1000:

                # EU距离小于100像素则旋转视角计算水平距离
                # 只有EU距离发现效果并不是很理想，加入y坐标与检测框底部距离进入if条件尝试
                Trees1D_ymin = []
                for xyxy, classitem, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
                    if int(classitem) == 0:
                        Trees1D_ymin.append(min([int(xyxy[1])]))
                closest_ytree = Trees1D_ymin[0]
                print(f"小于150会执行二次判断: {abs(closest_ytree)}")

                # 移动鼠标
                win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, target_x - current_x, (target_y - current_y) + 250, 0,
                                     0)
                # 前进
                win32api.keybd_event(0x57, 0, 0, 0)
                time.sleep(1)
                win32api.keybd_event(0x57, 0, win32con.KEYEVENTF_KEYUP, 0)

                # 截图（回到循环开始）
                win32api.keybd_event(0x54, 0, 0, 0)
                time.sleep(0.1)
                win32api.keybd_event(0x54, 0, win32con.KEYEVENTF_KEYUP, 0)

                if distance2D_to_tree < 300 and abs(closest_ytree) < 200:
                    del Trees1D_ymin[:]

                    # 将视角向右旋转到与树平行
                    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, screen_width // 2, 0, 0, 0)
                    time.sleep(0.5)

                    win32api.keybd_event(0x41, 0, 0, 0)
                    time.sleep(1)
                    win32api.keybd_event(0x41, 0, win32con.KEYEVENTF_KEYUP, 0)

                    try:
                        with mss.mss() as sct:
                            screenshot_1D_path = f"{SCREENSHOT_1D_PATH}/screenshot_{int(time.time())}.png"
                            processed_img, screenshot_1D_path = take_1Dscreenshot(sct, screenshot_1D_path)
                            print(f"1DScreenshot saved to {screenshot_1D_path}")
                            try:
                                with Image.open(screenshot_1D_path) as img:
                                    print("1DScreenshot saved successfully.")
                            except Exception as e:
                                print(f"Error saving 1Dscreenshot: {e}")
                    except Exception as e:
                        print(f"Error taking 1Dscreenshot: {e}")

                    # 进行模型推理
                    try:
                        print(f"Loading image from {screenshot_1D_path}")
                        results1D = model(screenshot_1D_path, **{'device': device})[0]
                        # print(f"Model output tensor shape: {results1D}")
                    except Exception as e:
                        print(f"Error: {e}")

                    # 横过来截图，看到了其他树并且被模型推理了怎么办呢？
                    # 可以找一个Y坐标差值最小的。计算并比较它与鼠标的Y坐标差，找到最小的之后计算它与鼠标的X坐标差
                    Trees1D_y = []
                    Trees1D_x = []
                    done = False

                    while not done:
                        for xyxy, classitem, conf in zip(results1D.boxes.xyxy, results1D.boxes.cls,
                                                         results1D.boxes.conf):
                            if int(classitem) == 0:
                                Trees1D_y.append(((int(xyxy[3]) + int(xyxy[1])) // 2))
                                Trees1D_x.append(((int(xyxy[2]) + int(xyxy[0])) // 2))
                                print(f"list of close trees y-axis: {Trees1D_y}")
                                print(f"list of close trees x-axis: {Trees1D_x}")
                                closest_tree_index = \
                                min(enumerate(Trees1D_y), key=lambda x: abs(x[1] - (screen_height // 2)))[0]
                                done = True  # 标志位为True，跳出while循环

                            if not Trees1D_y:
                                win32api.keybd_event(0x41, 0, 0, 0)
                                time.sleep(0.25)
                                win32api.keybd_event(0x41, 0, win32con.KEYEVENTF_KEYUP, 0)
                                try:
                                    time.sleep(1)
                                    with mss.mss() as sct:
                                        screenshot_1D_path = f"{SCREENSHOT_1D_PATH}/screenshot_{int(time.time())}.png"
                                        processed_img, screenshot_1D_path = take_1Dscreenshot(sct, screenshot_1D_path)
                                        print(f"1DScreenshot saved to {screenshot_1D_path}")
                                        try:
                                            with Image.open(screenshot_1D_path) as img:
                                                print("1DScreenshot saved successfully.")
                                        except Exception as e:
                                            print(f"Error saving 1Dscreenshot: {e}")
                                except Exception as e:
                                    print(f"Error taking 1Dscreenshot: {e}")

                                try:
                                    print(f"Loading image from {screenshot_1D_path}")
                                    results1D = model_Cut(screenshot_1D_path, **{'device': device})[0]
                                    # print(f"Model output tensor shape: {results1D}")
                                    done = False  # 标志位重置为False，继续循环
                                    break
                                except Exception as e:
                                    print(f"Error: {e}")

                        if done:  # 如果标志位为True，退出while循环
                            break
                            # # 回正
                            # win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, -(screen_width // 2), 0, 0, 0)
                            # time.sleep(0.5)
                            # # 回到循环开头
                            # win32api.keybd_event(0x54, 0, 0, 0)
                            # time.sleep(0.1)
                            # win32api.keybd_event(0x54, 0, win32con.KEYEVENTF_KEYUP, 0)

                    # 使用近距离树木模型推理
                    try:
                        # print(f"Loading image from {screenshot_path}")
                        results_Cut = model_Cut(screenshot_1D_path, **{'device': device})[0]
                        # print(f"Model output tensor shape: {results_Cut}")
                    except Exception as e:
                        print(f"Error: {e}")

                    Cutable_Trees = []
                    for xyxy, classitem, conf in zip(results_Cut.boxes.xyxy, results_Cut.boxes.cls, results_Cut.boxes.conf):
                        if int(classitem) == 0:
                            Cutable_Trees.append(1)

                    print(f"三次判断的条件需要小于500: {abs((Trees1D_x[closest_tree_index]) - (screen_width // 2))}")

                    if abs((Trees1D_x[closest_tree_index]) - (screen_width // 2)) < 500:
                        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                        time.sleep(2)
                        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                        time.sleep(2)
                        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)
                        win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)
                        time.sleep(2)
                        del Trees1D_y[:]
                        del Trees1D_x[:]
                        del Cutable_Trees[:]
                        # 回正
                        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, -(screen_width // 2), 0, 0, 0)

                        # 随机选择一个移动方式
                        move_name, move_params = random.choice(list(moves.items()))

                        # 根据选择的移动方式移动视角
                        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, move_params["x_offset"], 0, 0, 0)

                        # 移动角色并等待一段时间
                        win32api.keybd_event(0x53, 0, 0, 0)
                        time.sleep(0.5)
                        win32api.keybd_event(0x53, 0, win32con.KEYEVENTF_KEYUP, 0)

                        win32api.keybd_event(0x57, 0, 0, 0)
                        time.sleep(move_params["time_interval"])
                        win32api.keybd_event(0x57, 0, win32con.KEYEVENTF_KEYUP, 0)
                        time.sleep(move_params["time_interval"])

                        # 截图(回到主循环）
                        time.sleep(1)
                        win32api.keybd_event(0x54, 0, 0, 0)
                        time.sleep(0.1)
                        win32api.keybd_event(0x54, 0, win32con.KEYEVENTF_KEYUP, 0)

                    else:
                        del Trees1D_y[:]
                        del Trees1D_x[:]
                        del Cutable_Trees[:]

                        # 回正
                        win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, -(screen_width // 2), 0, 0, 0)
                        time.sleep(0.5)

                        # 截图（回到起点）
                        win32api.keybd_event(0x54, 0, 0, 0)
                        time.sleep(0.1)  # 等待 0.1 秒
                        win32api.keybd_event(0x54, 0, win32con.KEYEVENTF_KEYUP, 0)

                time.sleep(0.5)
                # 回到起点
                win32api.keybd_event(0x54, 0, 0, 0)
                time.sleep(0.1)  # 等待 0.1 秒
                win32api.keybd_event(0x54, 0, win32con.KEYEVENTF_KEYUP, 0)

            else:
                print("No Trees found(too far)")

        else:
            print("No Tress found")



    except Exception as e:
        print(f"Error1: {e}")

listener.stop()


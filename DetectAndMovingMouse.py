import math

import ctypes

import mss.tools
import torch
import pyautogui
import time
import os
import ctypes
import sys
from pynput.mouse import Controller
from pynput.keyboard import Listener, KeyCode
from torchvision import transforms
from PIL import Image


if not ctypes.windll.shell32.IsUserAnAdmin():
    # 如果当前用户不是管理员，则以管理员身份重新启动该进程
    ctypes.windll.shell32.ShellExecuteW(None, "runas", sys.executable, __file__, None, 1)
    sys.exit()

# 定义屏幕宽高
SCREEN_WIDTH = 2560
SCREEN_HEIGHT = 1600


# 定义截图路径和模型路径
SCREENSHOT_PATH = "random"
MODEL_PATH = "runs/detect/train2/weights/best.pt"

if not os.path.exists(SCREENSHOT_PATH):
    os.makedirs(SCREENSHOT_PATH)


# 定义树木类
class Tree:
    def __init__(self, xmin, ymin, xmax, ymax, confidence):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.confidence = confidence

    def get_center(self):
        x_center = int(self.xmin + (self.xmax - self.xmin) / 2)
        y_center = int(self.ymin + (self.ymax - self.ymin) / 2)
        return x_center, y_center

    def get_distance_to(self, x, y):
        return math.sqrt((self.get_center()[0] - x) ** 2 + (self.get_center()[1] - y) ** 2)


# 定义屏幕截图方法
def take_screenshot(sct, screenshot_path):
    try:
        img = sct.grab({"top": 0, "left": 0, "width": SCREEN_WIDTH, "height": SCREEN_HEIGHT})
        mss.tools.to_png(img.rgb, img.size, output=screenshot_path)


        # 读取刚刚截取的图片
        """img = Image.open(screenshot_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        # Resize and normalize the input image
        transform = transforms.Compose([
            transforms.Resize((640, 640)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        processed_img = transform(img)
        processed_img = processed_img.unsqueeze(0).repeat(16, 1, 1, 1).to(device)"""

        # img = Image.open(screenshot_path).convert("RGB")
        # # Resize and normalize the input image
        # transform = transforms.Compose([
        #     transforms.Resize((640, 640)),
        #     transforms.ToTensor(),
        #     #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        # ])
        # processed_img = transform(img)
        # processed_img = processed_img.unsqueeze(0).to(device)
        # print(f"Processed screenshot tensor shape: {processed_img.shape}")
        #
        #
        #
        # # 将预处理后的图像转换为 PIL.Image 类型并保存到文件
        # img_pil = transforms.ToPILImage()(processed_img.squeeze(0).to(device))
        # processed_screenshot_path = f"{SCREENSHOT_PATH}/screenshot_processed_{int(time.time())}.png"
        # img_pil.save(processed_screenshot_path)
        # print(f"Processed screenshot saved to {processed_screenshot_path}")
        # processed_img = processed_img.to(device)
        #
        # return processed_img, screenshot_path
        return img, screenshot_path


    except Exception as e:
        print(f"Error taking screenshot: {e}")


# 加载模型
print("Loading model,please wait")
try:
    model_dict = torch.load(MODEL_PATH, map_location='cpu')
    print("Model load successfully,press t to start and press f to quit")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# 将模型字典转换为 nn.Module 实例
model = model_dict['model'].float().fuse().eval()

# 移动模型到 CPU 或 GPU
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)

# 初始化鼠标
mouse = Controller()

# 初始化键盘监听器
exit_key = KeyCode(char='f')
pressed_keys = set()
start_key = KeyCode(char='t')


# 定义键盘按下事件
def on_press(key):
    if key == exit_key:
        print('Exiting...')
        return False
    if key == start_key:
        print('Starting detection...')
        # 清空按键集合，防止按键重复
        pressed_keys.clear()
        # 加入T键
        pressed_keys.add(key)

# 定义键盘释放事件
def on_release(key):
    if key in pressed_keys:
        pressed_keys.remove(key)

listener = Listener(on_press=on_press, on_release=on_release)
listener.start()

while True:

    try:
        # 检查是否按下退出键
        if exit_key in pressed_keys:
            print('Exiting...')
            sys.exit(0)
            break
        # 检查是否按下开始键
        if start_key in pressed_keys:
            print('Starting detection...')
            while start_key in pressed_keys:
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
                    print(f"Error taking screenshot:{e}")

        # 进行模型推理
        try:
            print(f"Loading image from {screenshot_path}")
            results = model(processed_img)
            print(f"Model output tensor shape: {results[0].shape}")
        except Exception as e:
            print(f"Error: {e}")

        # 获取树木列表（类别编号是classitem,0是树1是旅行者）
        Trees = []
        for i in range(results[0].shape[0]):
            classitem = results[0][i][-1].detach().cpu().numpy()[0]
            conf = results[0][i][-2].detach().cpu().numpy()[0]
            print(f"Class: {classitem}, Confidence: {conf}")
            if classitem == 0 and conf > 0.5:
                xmin, ymin, xmax, ymax = results[0][i][2:6].detach().cpu().numpy()
                Trees.append(Tree(xmin, ymin, xmax, ymax, conf))

        if len(Trees) > 0:
            # 找到距离最近的树木
            closest_tree = min(Trees, key=lambda e: e.get_distance_to(*mouse.position))

            # 计算距离
            distance_to_tree = closest_tree.get_distance_to(*mouse.position).item()

            # 如果树木距离鼠标坐标小于150则自动进行瞄准
            if distance_to_tree < 1000:
                # 移动并点击鼠标
                pyautogui.click(closest_tree.get_center()[0], closest_tree.get_center()[1])
                print(f"Closest Tree: {closest_tree.get_center()} ({distance_to_tree:.2f} pixels)")
        else:
            print("No Trees found")

        # 加入短暂的延迟，以避免cpu被过度占用
        time.sleep(0.1)
    except Exception as e:
        print(f"Error: {e}")

listener.stop()


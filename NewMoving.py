# 这里是导入依赖，需要这些库
# import ctypes
import math

import mss.tools
import torch
import pyautogui
from pynput.mouse import Controller
from ultralytics import YOLO

# 传入两个坐标点，计算直线距离的
class Line:
    def __init__(self, start_x, start_y, end_x, end_y):
        self.start_x = start_x
        self.start_y = start_y
        self.end_x = end_x
        self.end_y = end_y

    def get_length(self):
        return math.sqrt(math.pow((self.start_x - self.end_x), 2) + math.pow((self.start_y - self.end_y), 2))


# 加载本地模型
device = torch.device("cpu")
model = YOLO('runs/detect/train2/weights/best.pt')
# 定义屏幕宽高
game_width = 1920
game_height = 1080

rect = (0, 0, game_width, game_height)
m = mss.mss()
mt = mss.tools

# 加载罗技鼠标驱动，驱动资源来自互联网
# driver = ctypes.CDLL('myProjects/logitech.driver.dll')
# ok = driver.device_open() == 1
# if not ok:
#    print('初始化失败, 未安装lgs/ghub驱动')


# 截图保存
def screen_record():
    img = m.grab(rect)
    mt.to_png(img.rgb, img.size, 6, "pics/gen.png")


# 这边就是开始实时进行游戏窗口推理了
# 无限循环 -> 截取屏幕 -> 推理模型获取到每个敌人坐标 -> 计算每个敌人中心坐标 -> 挑选距离准星最近的敌人 -> 则控制鼠标移动到敌人的身体或者头部
while True:
    # 截取屏幕
    screen_record()
    # 使用模型
    # model = model.to(device)
    # 开始推理
    results = model('pics/gen.png', **{'device':'cpu'})[0]
    # 过滤模型
    newlist = []
    for xyxy, classitem, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        if int(classitem) == 0 and float(conf) > 0.5:
            newlist.append([int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]), float(conf)])
    # 循环遍历每个敌人的坐标信息传入距离计算方法获取每个敌人距离鼠标的距离
    if len(newlist) > 0:
        print('newlist:', newlist)
        # 存放距离数据
        cdList = []
        xyList = []
        for listItem in newlist:
            # 当前遍历的人物中心坐标
            xindex = int(listItem[2] - (listItem[2] - listItem[0]) / 2)
            yindex = int(listItem[3] - (listItem[3] - listItem[1]) * 2 / 3)
            mouseModal = Controller()
            x, y = mouseModal.position
            L1 = Line(x, y, xindex, yindex)
            print(int(L1.get_length()), x, y, xindex, yindex)
            # 获取到距离并且存放在cdList集合中
            cdList.append(int(L1.get_length()))
            xyList.append([xindex, yindex, x, y])
        # 这里就得到了距离最近的敌人位置了
        minCD = min(cdList)
        # 如果敌人距离鼠标坐标小于150则自动进行瞄准，这里可以改大改小，小的话跟枪会显得自然些
        if minCD < 150:
            for cdItem, xyItem in zip(cdList, xyList):
                if cdItem == minCD:
                    print(cdItem, xyItem)
                    # 使用驱动移动鼠标
                    pyautogui.moveTo(int(xyItem[0] - xyItem[2]),
                                 int(xyItem[1] - xyItem[3]), True)
                break

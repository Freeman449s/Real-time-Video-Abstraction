"""
功能性函数
"""

import math
import numpy as np
import cv2 as cv
from enum import Enum
from skimage import color


def gaussian(t, sigma) -> float:
    """
    期望为0的高斯函数。由于系数部分会在取后续平均时直接略去，因而只计算e的乘方。\n
    :param t: e的乘方的次数的分子
    :param sigma: 高斯函数的标准差
    :return: 高斯函数值
    """
    return math.e ** (-t / (2 * (sigma ** 2)))


def lab2bgr(lab: np.ndarray) -> np.ndarray:
    """
    将标准值域的lab图像转为适合CV显示的BGR图像\n
    :param lab: 标准值域LAB图像
    :return: BGR图像
    """
    rgb = color.lab2rgb(lab) * 255  # CV使用不标准的lab值域，不能使用CV的转换函数；lab是float形式，rgb将被归一化到[0,1]，需乘以255
    rgb = rgb.astype(np.uint8)  # 转为uint8
    bgr = cv.cvtColor(rgb, cv.COLOR_RGB2BGR)
    return bgr


def calcGradient(lab: np.ndarray, x: int, y: int) -> float:
    """
    计算lab图像在(x,y)位置处的亮度梯度（近似值）\n
    :param lab: lab图像
    :param x: 横坐标
    :param y: 纵坐标
    :return: lab图像在(x,y)位置处的亮度梯度（近似值）
    """
    if (y + 1 < lab.shape[0]):
        yDiff = lab[y + 1][x][0] - lab[y][x][0]
    else:
        yDiff = 0
    if (x + 1 < lab.shape[1]):
        xDiff = lab[y][x + 1][0] - lab[y][x][0]
    else:
        xDiff = 0
    return abs(yDiff) + abs(xDiff)

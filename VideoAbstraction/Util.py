"""
功能性函数
"""

import math, numpy as np, cv2 as cv
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

"""
高斯差分边缘检测
"""

import math
import numpy as np

from Util import gaussian

TAU = 0.98
PHI_E = 5  # 控制阶跃函数的梯度


def DoG(lab: np.ndarray, sigma_e, windowSize: int = 5) -> np.ndarray:
    """
    DoG边缘检测，返回边缘图\n
    :param lab: lab图像
    :param sigma_e: 控制边缘检测空间尺度
    :param windowSize: 高斯滤波的窗口大小，默认为5
    :return: lab形式的边缘图
    """
    edge = np.zeros(lab.shape)
    edge[:, :, 0] = 100
    for y in range(windowSize // 2, lab.shape[0] - windowSize // 2):
        for x in range(windowSize // 2, lab.shape[1] - windowSize // 2):
            S_sigma_e = blurFunc(lab, x, y, sigma_e, windowSize)
            S_sigma_r = blurFunc(lab, x, y, sigma_e * 1.264911, windowSize)
            if (S_sigma_e - TAU * S_sigma_r > 0):
                D = 1
            else:
                D = 1 + math.tanh(PHI_E * (S_sigma_e - TAU * S_sigma_r))
            edge[y][x][0] = D * 100
    return edge


def blurFunc(lab: np.ndarray, x: int, y: int, sigma_e, windowSize: int = 5) -> float:
    """
    模糊函数S\n
    :param lab: lab图像
    :param x: 窗口中心的横坐标
    :param y: 窗口中心的纵坐标
    :param sigma_e: 控制空间尺度
    :param windowSize: 窗口大小，默认为5
    :return: 模糊函数S的值
    """
    sum = 0
    for j in range(y - windowSize // 2, y + windowSize // 2 + 1):
        for i in range(x - windowSize // 2, x + windowSize // 2 + 1):
            sum += lab[j][i][0] * gaussian((j - y) ** 2 + (i - x) ** 2, sigma_e)
    return sum / (2 * math.pi * (sigma_e ** 2))


def overlayEdges(lab: np.ndarray, edge: np.ndarray) -> np.ndarray:
    """
    复制lab图像，将边缘叠加到图像的副本上\n
    :param lab: lab图像
    :param edge: lab空间下的边缘图
    :return: 叠加边缘的图像副本
    """
    ret = lab.copy()
    for y in range(0, ret.shape[0]):
        for x in range(0, ret.shape[1]):
            if (edge[y][x][0] < 30):  # 此处检测出边缘
                ret[y][x][0] = edge[y][x][0]
    return ret

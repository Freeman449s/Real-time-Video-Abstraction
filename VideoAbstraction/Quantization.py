"""
量化
"""

import math
import numpy as np

from Util import calcGradient

N_BINS = 8  # 桶数量


def quantize(lab: np.ndarray) -> np.ndarray:
    """
    创建lab图像的副本，然后对副本的亮度进行量化\n
    :param lab: lab图像
    :return: lab图像副本的量化图像
    """
    ret = lab.copy()
    for y in range(0, ret.shape[0]):
        for x in range(0, ret.shape[1]):
            ret[y][x][0] = QFunc(lab, x, y)
    return ret


def QFunc(lab: np.ndarray, x: int, y: int) -> float:
    """
    量化函数，计算lab图像上某点亮度的量化值\n
    :param lab: lab图像
    :param x: 横坐标
    :param y: 纵坐标
    :return: lab图像上(x,y)位置处亮度的量化值
    """
    luminance = lab[y][x][0]
    qNearest = findQNearest(luminance, N_BINS)
    binWidth = 100 / N_BINS
    gradient = calcGradient(lab, x, y)
    phi_q = calc_phi_q(gradient)
    return qNearest + binWidth / 2 * math.tanh(phi_q * (luminance - qNearest))


def calc_phi_q(gradient: float) -> float:
    """
    根据梯度计算phi_q. 梯度到phi_q的映射采用论文建议的映射，将[0,2]范围内的梯度映射到[3,14]范围内的phi_q.\n
    :param gradient: lab图像在某点的亮度梯度
    :return: 该点的phi_q
    """
    if (gradient > 2):
        return 14
    return 11 / 2 * gradient + 3


def findQNearest(luminance: float, nBins: int):
    """
    返回最近的桶边界\n
    :param luminance: 自变量，应为lab图像的亮度值
    :return: 距离x最近的桶边界
    """
    binWidth = 100 / nBins
    qNearest = int((luminance + binWidth / 2) / binWidth) * binWidth
    if (qNearest > 100):
        qNearest = 100
    return qNearest

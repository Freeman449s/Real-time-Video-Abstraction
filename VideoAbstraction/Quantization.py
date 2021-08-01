"""
量化
"""

import math, multiprocessing
import numpy as np

from Util import calcGradient
from Parallel import calcBoundaries, Branch

N_BINS = 8  # 桶数量


def quantize(lab: np.ndarray, multiProcess: bool = True) -> np.ndarray:
    """
    创建lab图像的副本，然后对副本的亮度进行量化\n
    :param lab: lab图像
    :return: lab图像副本的量化图像
    """
    if not multiProcess:
        return branchQuantize(lab, 0, lab.shape[1], 0, lab.shape[0])
    else:
        ret = lab.copy()
        nSegments = int(multiprocessing.cpu_count() ** 0.5) + 1  # 计算一边上的分段数
        xStartList, xEndList, yStartList, yEndList = calcBoundaries(ret.shape, nSegments, 1)

        pool = multiprocessing.Pool()
        manager = multiprocessing.Manager()
        queue = manager.Queue()
        for i in range(0, nSegments):
            for j in range(0, nSegments):
                pool.apply_async(branchQuantize,
                                 args=(lab, xStartList[i], xEndList[i], yStartList[j], yEndList[j], queue))
        pool.close()
        pool.join()
        # 将队列中的数据复制回ret
        while not queue.empty():
            branch = queue.get()
            ret[branch.yStart:branch.yEnd, branch.xStart:branch.xEnd] = branch.lab
        return ret


def branchQuantize(lab: np.ndarray, xStart: int, xEnd: int, yStart: int, yEnd: int, queue=None) -> np.ndarray:
    """
    对lab图像的分支进行量化\n
    :param lab: lab图像
    :param xStart: 分支横坐标的下界
    :param xEnd: 分支横坐标的上界（不包含）
    :param yStart: 分支纵坐标的下界
    :param yEnd: 分支纵坐标的上界（不包含）
    :param queue: 由服务进程管理的队列，用于向主进程传递对象
    :return: 分支的量化图像
    """
    ret = lab.copy()
    for y in range(yStart, yEnd):
        for x in range(xStart, xEnd):
            ret[y][x][0] = QFunc(lab, x, y)
    if queue:
        queue.put(Branch(ret[yStart:yEnd, xStart:xEnd], xStart, xEnd, yStart, yEnd))
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

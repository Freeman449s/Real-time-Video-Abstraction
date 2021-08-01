"""
高斯差分边缘检测
"""

import math, multiprocessing
import numpy as np

from Util import gaussian
from Parallel import Branch, calcBoundaries
from Exception import IllegalArgumentException
from scipy import signal

TAU = 0.98
PHI_E = 5  # 控制阶跃函数的梯度


def DoG(lab: np.ndarray, sigma_e: float, windowSize: int = 5, multiProcess: bool = True) -> np.ndarray:
    """
    DoG边缘检测，返回边缘图\n
    :param lab: lab图像
    :param sigma_e: 控制边缘检测空间尺度
    :param windowSize: 高斯滤波的窗口大小，默认为5
    :param multiProcess: 是否使用多进程加速
    :return: lab形式的边缘图
    """
    if not multiProcess:
        return branchDoG(lab, windowSize // 2, lab.shape[1] - windowSize // 2,
                         windowSize // 2, lab.shape[0] - windowSize // 2, sigma_e, windowSize)
    else:
        edge = np.zeros(lab.shape)
        edge[:, :, 0] = 100
        nSegments = int(multiprocessing.cpu_count() ** 0.5) + 1
        xStartList, xEndList, yStartList, yEndList = calcBoundaries(edge.shape, nSegments, windowSize)

        pool = multiprocessing.Pool()
        manager = multiprocessing.Manager()
        queue = manager.Queue()
        for i in range(0, nSegments):
            for j in range(0, nSegments):
                pool.apply_async(branchDoG,
                                 args=(lab, xStartList[i], xEndList[i], yStartList[j], yEndList[j],
                                       sigma_e, windowSize, queue))
        pool.close()
        pool.join()

        while not queue.empty():
            branch = queue.get()
            edge[branch.yStart:branch.yEnd, branch.xStart:branch.xEnd] = branch.lab

        return edge


def branchDoG(lab: np.ndarray, xStart: int, xEnd: int, yStart: int, yEnd: int, sigma_e: float, windowSize: int = 5,
              queue=None) -> np.ndarray:
    """
    使用多进程，对图像分支进行DoG边缘检测。结果通过队列返回（如有）。\n
    :param lab: lab图像
    :param xStart: 分支横坐标的下界
    :param xEnd: 分支横坐标的上界（不包含）
    :param yStart: 分支纵坐标的下界
    :param yEnd: 分支纵坐标的上界（不包含）
    :param sigma_e: 控制边缘检测空间尺度
    :param windowSize: 窗口大小，默认为5
    :param queue: 由服务进程管理的队列，用于向主进程传递对象
    :return: 边缘图像
    """
    edge = np.zeros(lab.shape)
    edge[:, :, 0] = 100
    for y in range(yStart, yEnd):
        for x in range(xStart, xEnd):
            S_sigma_e = blurFunc(lab, x, y, sigma_e, windowSize)
            S_sigma_r = blurFunc(lab, x, y, sigma_e * 1.264911, windowSize)  # 1.264911 = 1.6^(1/2)
            if (S_sigma_e - TAU * S_sigma_r > 0):
                D = 1
            else:
                D = 1 + math.tanh(PHI_E * (S_sigma_e - TAU * S_sigma_r))
            edge[y][x][0] = D * 100
    if queue:
        queue.put(Branch(edge[yStart:yEnd, xStart:xEnd], xStart, xEnd, yStart, yEnd))
    return edge


def blurFunc(lab: np.ndarray, x: int, y: int, sigma_e: float, windowSize: int = 5) -> float:
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
            ret[y][x][0] = min(ret[y][x][0], edge[y][x][0])
    return ret


def DoGUsingScipy(lab: np.ndarray, sigma_e: float, windowSize: int = 5) -> np.ndarray:
    """
    使用scipy.signal.convolve()计算DoG\n
    :param lab: lab图像
    :param sigma_e: 控制边缘检测空间尺度
    :param windowSize: 窗口大小，默认为5
    :return: 边缘图像
    """
    conSigmaE = signal.convolve(lab[:, :, 0], calcKernel(sigma_e, windowSize), mode="same")  # mode取"same"时，输出与in1尺寸相同
    conSigmaR = signal.convolve(lab[:, :, 0], calcKernel(sigma_e * 1.264911, windowSize),
                                mode="same")  # 1.264911 = 1.6^(1/2)

    edge = np.zeros(lab.shape)
    edge[:, :, 0] = 100
    for y in range(0, edge.shape[0]):
        for x in range(0, edge.shape[1]):
            S_sigma_e = conSigmaE[y][x]
            S_sigma_r = conSigmaR[y][x]
            if (S_sigma_e - TAU * S_sigma_r > 0):
                D = 1
            else:
                D = 1 + math.tanh(PHI_E * (S_sigma_e - TAU * S_sigma_r))
            edge[y][x][0] = D * 100
    return edge


def calcKernel(sigma_e: float, windowSize: int = 5) -> np.ndarray:
    """
    计算DoG边缘检测的卷积核\n
    :param windowSize: 窗口尺寸，默认为5
    :param sigma_e: 控制边缘检测空间尺度
    :return: 卷积核
    """
    if (windowSize // 2 == 0):
        raise IllegalArgumentException("Window size must be odd.")
    kernel = np.zeros((windowSize, windowSize))
    for y in range(-windowSize // 2, windowSize // 2 + 1):
        for x in range(-windowSize // 2, windowSize // 2 + 1):
            kernel[y + windowSize // 2][x + windowSize // 2] = gaussian((y ** 2 + x ** 2), sigma_e)
    kernel /= (2 * math.pi * (sigma_e ** 2))
    return kernel

"""
双边滤波
"""

import math, multiprocessing
import numpy as np

from Util import gaussian
from Parallel import calcBoundaries, Branch


def bilateral(lab: np.ndarray, sigma_d, sigma_r, windowSize: int = 5, multiProcess: bool = True) -> np.ndarray:
    """
    建立图像的副本，然后双边滤波。\n
    :param lab: lab图像
    :param sigma_d: 空间域标准差
    :param sigma_r: 像素域标准差
    :param windowSize: 窗口大小，默认为5
    :return: 经双边滤波的图像
    """
    ret = lab.copy()
    if (not multiProcess):
        branchFilter(ret, windowSize // 2, ret.shape[1] - windowSize // 2,
                     windowSize // 2, ret.shape[0] - windowSize // 2, sigma_d, sigma_r, windowSize)
    else:
        nSegments = int(multiprocessing.cpu_count() ** 0.5) + 1  # 计算一边上的分段数
        xStartList, xEndList, yStartList, yEndList = calcBoundaries(ret.shape, nSegments, windowSize)

        pool = multiprocessing.Pool()
        manager = multiprocessing.Manager()
        queue = manager.Queue()
        for i in range(0, nSegments):
            for j in range(0, nSegments):
                pool.apply_async(branchFilter,
                                 args=(ret, xStartList[i], xEndList[i], yStartList[j], yEndList[j],
                                       sigma_d, sigma_r, windowSize, queue))
        pool.close()
        pool.join()
        # 将队列中的数据复制回ret
        while not queue.empty():
            branch = queue.get()
            ret[branch.yStart:branch.yEnd, branch.xStart:branch.xEnd] = branch.lab
    return ret


def branchFilter(lab: np.ndarray, xStart: int, xEnd: int, yStart: int, yEnd: int,
                 sigma_d, sigma_r, windowSize: int = 5, queue=None) -> None:
    """
    使用多进程，为图像分支进行双边滤波。将直接在原图像上操作。\n
    :param lab: lab图像。如果使用多进程，在传入时会复制一份。
    :param xStart: 分支横坐标的下界
    :param xEnd: 分支横坐标的上界（不包含）
    :param yStart: 分支纵坐标的下界
    :param yEnd: 分支纵坐标的上界（不包含）
    :param sigma_d: 空间域标准差
    :param sigma_r: 像素域标准差
    :param windowSize: 窗口大小，默认为5
    :param queue: 由服务进程管理的队列，用于向主进程传递对象
    :return: 无返回值
    """
    for y in range(yStart, yEnd):
        for x in range(xStart, xEnd):
            weightSum = 0
            pixelSum = 0
            for j in range(y - windowSize // 2, y + windowSize // 2 + 1):
                for i in range(x - windowSize // 2, x + windowSize // 2 + 1):
                    weight = gaussian((j - y) ** 2 + (i - x) ** 2, sigma_d) * \
                             gaussian((lab[j][i][0] - lab[y][x][0]) ** 2, sigma_r)
                    weightSum += weight
                    pixelSum += lab[j][i][0] * weight
            lab[y][x][0] = pixelSum / weightSum
    if queue:
        queue.put(Branch(lab[yStart:yEnd, xStart:xEnd], xStart, xEnd, yStart, yEnd))  # 注意ndarray的切片方式[a:b,c:d]

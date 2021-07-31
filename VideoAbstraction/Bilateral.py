"""
双边滤波
"""

import math, multiprocessing, numpy as np

MULTI_THREADING = False


class Branch:
    """
    图像分支。用于子进程向主进程传递数据。
    """

    def __init__(self, lab: np.ndarray, xStart: int, xEnd: int, yStart: int, yEnd: int):
        self.lab = lab
        self.xStart = xStart
        self.xEnd = xEnd
        self.yStart = yStart
        self.yEnd = yEnd


def gaussian(t, sigma) -> float:
    """
    期望为0的高斯函数。由于系数部分会在取后续平均时直接略去，因而只计算e的乘方。\n
    :param t: e的乘方的次数的分子
    :param sigma: 高斯函数的标准差
    :return: 高斯函数值
    """
    return math.e ** (-t / (2 * (sigma ** 2)))


def branchFilter(lab: np.ndarray, xStart: int, xEnd: int, yStart: int, yEnd: int,
                 sigma_d, sigma_r, windowSize: int = 3, queue=None) -> None:
    """
    使用多进程，为图像分支进行双边滤波。将直接在原图像上操作。\n
    :param lab: lab图像
    :param xStart: 分支横坐标的下界
    :param xEnd: 分支横坐标的上界（不包含）
    :param yStart: 分支纵坐标的下界
    :param yEnd: 分支纵坐标的上界（不包含）
    :param sigma_d: 空间域标准差
    :param sigma_r: 像素域标准差
    :param windowSize: 窗口大小
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


def bilateral(lab: np.ndarray, sigma_d, sigma_r, windowSize: int = 3) -> np.ndarray:
    """
    建立图像的副本，然后双边滤波。\n
    :param lab: lab图像
    :param sigma_d: 空间域标准差
    :param sigma_r: 像素域标准差
    :param windowSize: 窗口大小
    :return: 经双边滤波的图像
    """
    ret = lab.copy()
    if (not MULTI_THREADING):
        branchFilter(ret, windowSize // 2, ret.shape[1] - windowSize // 2,
                     windowSize // 2, ret.shape[0] - windowSize // 2, sigma_d, sigma_r, windowSize)
    else:
        nSegments = int(multiprocessing.cpu_count() ** 0.5) + 1  # 计算一边上的分段数
        xSegmentLen = ret.shape[1] // nSegments
        ySegmentLen = ret.shape[0] // nSegments
        # 计算各分支边界
        xStartList = [windowSize // 2]
        xEndList = []
        yStartList = [windowSize // 2]
        yEndList = []
        for i in range(1, nSegments):
            xStartList.append(i * xSegmentLen)
            xEndList.append(i * xSegmentLen)
            yStartList.append(i * ySegmentLen)
            yEndList.append(i * ySegmentLen)
        xEndList.append(ret.shape[1] - windowSize // 2)
        yEndList.append(ret.shape[0] - windowSize // 2)

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

import numpy as np


class Branch:
    """
    图像分支。用于并行运算中，子进程向主进程传递数据。
    """

    def __init__(self, lab: np.ndarray, xStart: int, xEnd: int, yStart: int, yEnd: int):
        self.lab = lab
        self.xStart = xStart
        self.xEnd = xEnd
        self.yStart = yStart
        self.yEnd = yEnd


def calcBoundaries(shape: tuple, nSegments: int, windowSize: int = 5) -> tuple:
    """
    根据每一边上的段数，将图像划分为若干子区域，方便并行计算。返回各个区域横纵坐标的上下边界组成的列表。\n
    :param shape: 原图像的尺寸。默认为3通道图像。
    :param nSegments: 每一边上分割的段数
    :param windowSize: 滤波器窗口大小，默认为5
    :return: 4个列表，分别表示各区域横坐标下界、横坐标上界、纵坐标下界、纵坐标上界
    """
    xSegmentLen = shape[1] // nSegments
    ySegmentLen = shape[0] // nSegments
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
    xEndList.append(shape[1] - windowSize // 2)
    yEndList.append(shape[0] - windowSize // 2)
    return xStartList, xEndList, yStartList, yEndList

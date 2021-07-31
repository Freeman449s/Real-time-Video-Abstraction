"""
备份不用的类和函数
"""

from threading import Thread
import math, numpy as np


class BilateralBranch(Thread):
    """
    对图像的一个分支做双边滤波
    """

    def __init__(self, lab: np.ndarray, xStart: int, xEnd: int, yStart: int, yEnd: int, sigma_d, sigma_r,
                 windowSize: int = 3):
        """
        构造函数\n
        :param lab: lab图像，将直接对此图像做修改
        :param xStart: 分支横坐标的下界
        :param xEnd: 分支横坐标的上界（不包含）
        :param yStart: 分支纵坐标的下界
        :param yEnd: 分支纵坐标的上界（不包含）
        :param sigma_d: 空间域标准差
        :param sigma_r: 像素域标准差
        """
        Thread.__init__(self)
        self.lab = lab
        self.xStart = xStart
        self.xEnd = xEnd
        self.yStart = yStart
        self.yEnd = yEnd
        self.sigma_d = sigma_d
        self.sigma_r = sigma_r
        self.windowSize = windowSize

    def run(self):
        for y in range(self.yStart, self.yEnd):
            for x in range(self.xStart, self.xEnd):
                weightSum = 0
                pixelSum = 0
                for j in range(y - self.windowSize // 2, y + self.windowSize // 2 + 1):
                    for i in range(x - self.windowSize // 2, x + self.windowSize // 2 + 1):
                        weight = gaussian((j - y) ** 2 + (i - x) ** 2, self.sigma_d) * \
                                 gaussian((self.lab[j][i][0] - self.lab[y][x][0]) ** 2, self.sigma_r)
                        weightSum += weight
                        pixelSum += self.lab[j][i][0] * weight
                self.lab[y][x][0] = pixelSum / weightSum


def gaussian(t, sigma) -> float:
    """
    期望为0的高斯函数。由于系数部分会在取后续平均时直接略去，因而只计算e的乘方。\n
    :param t: e的乘方的次数的分子
    :param sigma: 高斯函数的标准差
    :return: 高斯函数值
    """
    return math.e ** (-t / (2 * (sigma ** 2)))

class ConversionType(Enum):
    UByte2Classic = 0
    Classic2UByte = 1


def valueRangeConversion(lab: np.ndarray, conversionType: ConversionType) -> np.ndarray:
    """
    将LAB从[0,255]转到经典取值范围，即L:[0,100]，(a,b):[-127,127]；或从经典取值范围转回uint8范围\n
    :param lab: LAB图像
    :param conversionType: 转换类型
    :return: 转换过取值范围的LAB图像
    """
    if conversionType == ConversionType.UByte2Classic:
        ret = lab.astype(np.float64)
        for y in range(0, ret.shape[0]):
            for x in range(0, ret.shape[1]):
                ret[y][x][0] = ret[y][x][0] / 255 * 100
                ret[y][x][1] = ret[y][x][1] - 128
                ret[y][x][2] = ret[y][x][2] - 128
    else:
        ret = np.zeros(lab.shape, np.uint8)
        for y in range(0, ret.shape[0]):
            for x in range(0, ret.shape[1]):
                ret[y][x][0] = lab[y][x][0] / 100 * 255
                ret[y][x][1] = lab[y][x][1] + 128
                ret[y][x][2] = lab[y][x][2] + 128
    return ret

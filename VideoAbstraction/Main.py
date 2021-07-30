import math, cv2 as cv, numpy as np
from enum import Enum

SIGMA_D = 3  # 双边滤波的空间域标准差
SIGMA_R = 4.25  # 双边滤波的时间域标准差


class ConversionType(Enum):
    UByte2Classic = 0
    Classic2UByte = 1


def main():
    img = cv.imread("Boat.jpg")
    cv.imshow("Original", img)
    lab = cv.cvtColor(img, cv.COLOR_BGR2Lab)  # OpenCV以BGR读取图像
    lab = valueRangeConversion(lab, ConversionType.UByte2Classic)  # [0,255]转经典取值范围
    bilateral_lab = bilateral(lab, SIGMA_D, SIGMA_R)
    for i in range(0, 3):
        bilateral_lab = bilateral(bilateral_lab, SIGMA_D, SIGMA_R)
    bilateral_lab = valueRangeConversion(bilateral_lab, ConversionType.Classic2UByte)
    bilateral_bgr = cv.cvtColor(bilateral_lab, cv.COLOR_Lab2BGR)
    cv.imshow("Bilateral", bilateral_bgr)
    cv.waitKey()
    cv.destroyAllWindows()
    pass


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
    for y in range(windowSize // 2, ret.shape[0] - windowSize // 2):
        for x in range(windowSize // 2, ret.shape[1] - windowSize // 2):
            weightSum = 0
            pixelSum = 0
            for j in range(y - windowSize // 2, y + windowSize // 2 + 1):
                for i in range(x - windowSize // 2, x + windowSize // 2 + 1):
                    weight = gaussian((j - y) ** 2 + (i - x) ** 2, sigma_d) * \
                             gaussian((ret[j][i][0] - ret[y][x][0]) ** 2, sigma_r)
                    weightSum += weight
                    pixelSum += ret[j][i][0] * weight
            ret[y][x][0] = pixelSum / weightSum
    return ret


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


def gaussian(t, sigma) -> float:
    """
    期望为0的高斯函数。由于系数部分会在取后续平均时直接略去，因而只计算e的乘方。\n
    :param t: e的乘方的次数的分子
    :param sigma: 高斯函数的标准差
    :return: 高斯函数值
    """
    return math.e ** (-t / (2 * (sigma ** 2)))


main()

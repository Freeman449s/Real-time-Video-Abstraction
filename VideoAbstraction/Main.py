"""
主模块，定义程序的入口
"""

import math, Bilateral, cv2 as cv, numpy as np
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
    bilateral_lab = Bilateral.bilateral(lab, SIGMA_D, SIGMA_R)
    for i in range(0, 3):
        bilateral_lab = Bilateral.bilateral(bilateral_lab, SIGMA_D, SIGMA_R)
    bilateral_lab = valueRangeConversion(bilateral_lab, ConversionType.Classic2UByte)
    bilateral_bgr = cv.cvtColor(bilateral_lab, cv.COLOR_Lab2BGR)
    cv.imshow("Bilateral", bilateral_bgr)
    cv.waitKey()
    cv.destroyAllWindows()
    pass


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


if __name__ == '__main__':
    main()

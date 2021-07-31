"""
主模块，定义程序的入口
"""

import math, Bilateral
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from Bilateral import bilateral
from DoG import DoG, overlayEdges
from skimage import color, io
from Util import lab2bgr
from Quantization import quantize

SIGMA_D = 3  # 双边滤波的空间域标准差
SIGMA_R = 4.25  # 双边滤波的时间域标准差


def main(filePath: str):
    postFix = filePath.split(".")[-1]
    fileName = filePath[:len(filePath) - len(postFix) - 1]
    img = io.imread(filePath)
    img_bgr = cv.cvtColor(img, cv.COLOR_RGB2BGR)  # CV与skimage的通道顺序不同
    cv.imshow("Original", img_bgr)
    cv.waitKey(10)  # 这一步是为了让图像显示出来
    lab = color.rgb2lab(img)

    # 第1次双边滤波
    bilateral_lab = bilateral(lab, SIGMA_D, SIGMA_R)

    # 边缘检测
    edge = DoG(bilateral_lab, 0.65)
    edge_bgr = lab2bgr(edge)
    cv.imshow("DoG", edge_bgr)
    cv.waitKey(10)

    # 第2~4次双边滤波
    for i in range(0, 3):
        bilateral_lab = bilateral(bilateral_lab, SIGMA_D, SIGMA_R)
    bilateral_bgr = lab2bgr(bilateral_lab)
    cv.imshow("Bilateral", bilateral_bgr)
    cv.waitKey(10)

    # 量化
    quantized = quantize(bilateral_lab)
    quantized_bgr = lab2bgr(quantized)
    cv.imshow("Quantized", quantized_bgr)
    cv.waitKey(10)

    # 边缘叠加
    resultImg = overlayEdges(quantized, edge)
    resultImg_bgr = lab2bgr(resultImg)
    cv.imshow("Result", resultImg_bgr)
    cv.imwrite(fileName + "-out." + postFix, resultImg_bgr)
    cv.waitKey()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main("Boat1024.jpg")

"""
主模块，定义程序的入口
"""

import math, Bilateral, time
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
SIGMA_E = 0.6  # 控制DoG边缘检测的空间尺度


def main(filePath: str):
    postFix = filePath.split(".")[-1]
    fileName = filePath[:len(filePath) - len(postFix) - 1]
    img = io.imread(filePath)
    img_bgr = cv.cvtColor(img, cv.COLOR_RGB2BGR)  # CV与skimage的通道顺序不同
    cv.imshow("Original", img_bgr)
    cv.waitKey(10)  # 这一步是为了让图像显示出来
    lab = color.rgb2lab(img)

    # 第1~2次双边滤波
    print("第1~2次双边滤波：")
    startTime = time.time()
    bilateral_lab = bilateral(lab, SIGMA_D, SIGMA_R)
    bilateral_lab = bilateral(bilateral_lab, SIGMA_D, SIGMA_R)
    endTime = time.time()
    deltaTime = round(endTime - startTime, 2)
    print("\t" + str(deltaTime) + "s")

    # 边缘检测
    print("边缘检测：")
    startTime = time.time()
    edge = DoG(bilateral_lab, SIGMA_E)
    edge_bgr = lab2bgr(edge)
    endTime = time.time()
    deltaTime = round(endTime - startTime, 2)
    print("\t" + str(deltaTime) + "s")
    cv.imshow("DoG", edge_bgr)
    cv.imwrite("DoG-" + str(SIGMA_E) + ".jpg", edge_bgr)
    cv.waitKey(10)

    # 第3~4次双边滤波
    print("第3~4次双边滤波：")
    startTime = time.time()
    for i in range(0, 2):
        bilateral_lab = bilateral(bilateral_lab, SIGMA_D, SIGMA_R)
    bilateral_bgr = lab2bgr(bilateral_lab)
    endTime = time.time()
    deltaTime = round(endTime - startTime, 2)
    print("\t" + str(deltaTime) + "s")
    cv.imshow("Bilateral", bilateral_bgr)
    cv.waitKey(10)

    # 量化
    print("量化：")
    startTime = time.time()
    quantized = quantize(bilateral_lab)
    quantized_bgr = lab2bgr(quantized)
    endTime = time.time()
    deltaTime = round(endTime - startTime, 2)
    print("\t" + str(deltaTime) + "s")
    cv.imshow("Quantized", quantized_bgr)
    cv.waitKey(10)

    # 边缘叠加
    print("边缘叠加：")
    startTime = time.time()
    resultImg = overlayEdges(quantized, edge)
    resultImg_bgr = lab2bgr(resultImg)
    endTime = time.time()
    deltaTime = round(endTime - startTime, 2)
    print("\t" + str(deltaTime) + "s")
    cv.imshow("Result", resultImg_bgr)
    cv.imwrite(fileName + "-out." + postFix, resultImg_bgr)
    cv.waitKey()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main("Boat512.jpg")

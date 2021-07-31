"""
主模块，定义程序的入口
"""

import math, Bilateral, cv2 as cv, numpy as np

import matplotlib.pyplot as plt

from Bilateral import bilateral
from DoG import DoG, overlayEdges
from skimage import color, io
from Util import lab2bgr

SIGMA_D = 3  # 双边滤波的空间域标准差
SIGMA_R = 4.25  # 双边滤波的时间域标准差


def main():
    img = io.imread("Boat512.jpg")
    img_bgr = cv.cvtColor(img, cv.COLOR_RGB2BGR)  # CV与skimage的通道顺序不同
    cv.imshow("Original", img_bgr)
    lab = color.rgb2lab(img)

    # 第1次双边滤波
    bilateral_lab = bilateral(lab, SIGMA_D, SIGMA_R)

    # 边缘检测
    edge = DoG(bilateral_lab)
    edge_bgr = lab2bgr(edge)
    cv.imshow("DoG", edge_bgr)

    # 第2~4次双边滤波
    # for i in range(0, 3):
    #     bilateral_lab = bilateral(bilateral_lab, SIGMA_D, SIGMA_R)

    # 色彩空间转换与展示
    bilateralWithEdge = overlayEdges(bilateral_lab, edge)
    bilateral_bgr = lab2bgr(bilateral_lab)
    bilateralWithEdge_bgr = lab2bgr(bilateralWithEdge)
    cv.imshow("Bilateral", bilateral_bgr)
    cv.imshow("Bilateral with Edge", bilateralWithEdge_bgr)
    cv.waitKey()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()

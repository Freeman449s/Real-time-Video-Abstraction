[![license](https://img.shields.io/badge/license-LGPL--2.1-orange)]()
[![Python](https://img.shields.io/badge/language-Python-blue)]()

# Real-time Video Abstraction

#### 介绍
SIGGRAPH 2006论文的Python实现。

算法流程如下：

1. 将图像从RGB空间转到Lab空间
2. 双边滤波
3. DoG边缘检测
4. 亮度通道量化
5. 边缘叠加

A Python implementation of the SIGGRAPH 2006 paper.

The procedure of the algorithm is as follows:

1. Convert the input image from RGB color space to Lab.
2. Bilateral filtering.
3. DoG edge detection.
4. Quantization of L channel.
5. Overlay the edges onto the quantized image.

项目简介：
本项目旨在探索和实现多种图像处理技术，包括特征提取、图像复原、图像均衡化等方法，为机器视觉和图像分析研究提供支持。

主要脚本：
1.HOG特征提取.py: 实现HOG特征的提取，适用于物体识别。
2.Harris角点检测.py: 进行角点检测，用于图像特征提取。
3.Hough Transform.py: 用于检测图像中的几何形状。
4.PCA.py: 实现主成分分析，常用于降维。
5.图像复原.py: 提供图像复原的方法。
6.频率域拉普拉斯.py: 在频率域上进行图像处理。
7.图像分割.py:提供图像分割功能，支持从图像中提取感兴趣区域。图像分割是目标检测、医学影像处理等领域的基础步骤。
8.图像滤波对比.py:比较不同滤波方法的效果，例如均值滤波、高斯滤波和中值滤波等。通过对比，分析各滤波方法在去噪和保留细节方面的性能差异。
9.傅里叶频谱图集逆变换.py:展示图像的频谱图并进行逆变换，帮助理解频域处理对图像的影响。频域分析对于噪声抑制和特征增强具有重要意义。
10.RGB与HSI空间图像均衡化.py:提供RGB和HSI颜色空间的直方图均衡化方法，用于改善图像的对比度和亮度，提升图像的视觉效果。
11.RGB和HSI分量图.py:将图像分解为RGB和HSI空间的各分量图，便于分析颜色信息在不同空间中的分布特点，为图像增强和分割提供支持。


使用说明：
1.确保已安装必须的Python库，例如OpenCV和NumPy。
2.运行各个.py脚本以执行相应的图像处理任务。
3.输出结果存储在相应的结果目录中。

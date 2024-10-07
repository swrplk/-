import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import os

# 设置 Matplotlib 绘图参数，使其支持中文显示，并确保负号显示正确
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为 SimHei，以支持中文
plt.rcParams['axes.unicode_minus'] = False  # 允许在坐标轴上显示负号

# 通用函数：保存图像、绘图和显示对比
def save_and_plot(original_img, processed_img, name, title):
    # 检查是否存在存储锐化图像的目录，如果不存在则创建
    if not os.path.exists('sharpened_images'):
        os.makedirs('sharpened_images')
    # 保存处理后的图像到 'sharpened_images' 目录中
    cv.imwrite(f'sharpened_images/{name}.png', processed_img)

    # 准备绘制对比图像
    # 将原始图像从 BGR 转换为 RGB，以便 matplotlib 正确显示颜色
    images = [cv.cvtColor(original_img, cv.COLOR_BGR2RGB), processed_img]
    titles = ['原始图像', title]  # 设置两个子图的标题，分别是原始图像和处理后图像

    # 使用 Matplotlib 并排绘制原始图像和处理后的图像
    for i in range(2):
        plt.subplot(1, 2, i + 1)  # 创建 1 行 2 列的子图，当前为第 i+1 个
        plt.imshow(images[i], cmap='gray' if i == 1 else None)  # 如果是灰度图则设置 cmap 为 gray
        plt.title(titles[i])  # 设置子图的标题
        plt.xticks([]), plt.yticks([])  # 隐藏坐标轴刻度
    plt.show()  # 显示图像

# Roberts 算子
def roberts():
    img = cv.imread('orange.jpg')  # 读取图像 'orange.jpg'
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 将图像转换为灰度图
    kernelx = np.array([[-1, 0], [0, 1]], dtype=int)  # 定义 Roberts 算子的 X 方向核
    kernely = np.array([[0, -1], [1, 0]], dtype=int)  # 定义 Roberts 算子的 Y 方向核

    # 使用 2D 卷积滤波器来计算 X 和 Y 方向的边缘
    x = cv.filter2D(gray, cv.CV_16S, kernelx)
    y = cv.filter2D(gray, cv.CV_16S, kernely)

    # 将结果取绝对值并转换为 8 位图像
    absX = cv.convertScaleAbs(x)
    absY = cv.convertScaleAbs(y)
    # 合并 X 和 Y 方向的边缘检测结果
    result = cv.addWeighted(absX, 0.5, absY, 0.5, 0)

    # 保存和显示处理后的图像
    save_and_plot(img, result, 'roberts_sharpened', 'Roberts 算子')

# Sobel 算子
def sobel_operator():
    img = cv.imread('orange.jpg')  # 读取图像 'orange.jpg'
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 转换为灰度图

    # 使用 Sobel 算子在 X 和 Y 方向进行边缘检测
    x = cv.Sobel(gray, cv.CV_16S, 1, 0)  # X 方向
    y = cv.Sobel(gray, cv.CV_16S, 0, 1)  # Y 方向

    # 将结果取绝对值并转换为 8 位图像
    absX = cv.convertScaleAbs(x)
    absY = cv.convertScaleAbs(y)
    # 合并 X 和 Y 方向的边缘检测结果
    result = cv.addWeighted(absX, 0.5, absY, 0.5, 0)

    # 保存和显示处理后的图像
    save_and_plot(img, result, 'sobel_sharpened', 'Sobel 算子')

# Prewitt 算子
def prewitt_operator():
    img = cv.imread('orange.jpg')  # 读取图像 'orange.jpg'
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 转换为灰度图

    # 定义 Prewitt 算子的 X 和 Y 方向核
    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)

    # 使用 2D 卷积滤波器计算 X 和 Y 方向的边缘
    x = cv.filter2D(gray, cv.CV_16S, kernelx)
    y = cv.filter2D(gray, cv.CV_16S, kernely)

    # 将结果取绝对值并转换为 8 位图像
    absX = cv.convertScaleAbs(x)
    absY = cv.convertScaleAbs(y)
    # 合并 X 和 Y 方向的边缘检测结果
    result = cv.addWeighted(absX, 0.5, absY, 0.5, 0)

    # 保存和显示处理后的图像
    save_and_plot(img, result, 'prewitt_sharpened', 'Prewitt 算子')

# Kirsch 算子
def kirsch_operator():
    img = cv.imread('orange.jpg')  # 读取图像 'orange.jpg'
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 转换为灰度图

    # 定义 Kirsch 算子的方向核，只有部分方向核以节省空间
    kirsch_kernels = [
        np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]], dtype=int),  # 核1
        np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]], dtype=int),  # 核2
        # 其他方向的核省略，以节省空间
    ]

    # 对每个方向核进行卷积操作，计算响应
    responses = [cv.filter2D(gray, cv.CV_16S, kernel) for kernel in kirsch_kernels]
    # 将响应取绝对值并转换为 8 位图像
    abs_responses = [cv.convertScaleAbs(response) for response in responses]
    # 取所有方向的最大值，得到最强的边缘响应
    result = np.max(abs_responses, axis=0)

    # 保存和显示处理后的图像
    save_and_plot(img, result, 'kirsch_sharpened', 'Kirsch 算子')

# Laplace 算子
def laplace_sharpening():
    img = cv.imread('orange.jpg')  # 读取图像 'orange.jpg'
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 转换为灰度图

    # 使用 Laplace 算子计算二阶导数，突出边缘
    laplacian = cv.Laplacian(gray, cv.CV_16S, ksize=3)
    # 将结果取绝对值并转换为 8 位图像
    result = cv.convertScaleAbs(laplacian)

    # 保存和显示处理后的图像
    save_and_plot(img, result, 'laplace_sharpened', 'Laplace 二阶锐化')

# 主程序入口
num = input("请选择功能（1-Roberts, 2-Sobel, 3-Prewitt, 4-Kirsch, 5-Laplace）：")

# 根据用户输入，调用相应的锐化算子函数
if num == '1':
    roberts()
elif num == '2':
    sobel_operator()
elif num == '3':
    prewitt_operator()
elif num == '4':
    kirsch_operator()
elif num == '5':
    laplace_sharpening()
else:
    # 输入无效时，打印错误提示
    print("无效输入，请输入1到5之间的数字。")

import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import os

# 设置 matplotlib 的显示字体和负号的显示方式
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用 SimHei 字体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号


# 通用函数：保存图像、绘图和显示对比
def save_and_plot(original_img, processed_img, name, title):
    # 创建保存目录，如果不存在则创建
    if not os.path.exists('sharpened_images'):
        os.makedirs('sharpened_images')

    # 保存锐化后的图像
    cv.imwrite(f'sharpened_images/{name}.png', processed_img)

    # 将处理后的图像转换为三通道，以便于拼接原图
    processed_img_color = cv.cvtColor(processed_img, cv.COLOR_GRAY2BGR)

    # 拼接原图和处理后的图像进行对比
    compare_image = np.hstack((original_img, processed_img_color))
    # 保存对比图像
    cv.imwrite(f'sharpened_images/compare_{name}.png', compare_image)

    # 绘制对比图像
    images = [cv.cvtColor(original_img, cv.COLOR_BGR2RGB), processed_img]  # 将原图转换为 RGB 格式
    titles = ['原始图像', title]  # 设置图像标题
    for i in range(2):
        plt.subplot(1, 2, i + 1)  # 创建 1 行 2 列的子图
        plt.imshow(images[i], cmap='gray' if i == 1 else None)  # 根据索引选择显示图像
        plt.title(titles[i])  # 设置标题
        plt.xticks([]), plt.yticks([])  # 隐藏坐标轴刻度
    plt.show()  # 显示对比图


# Roberts 算子锐化函数
def roberts():
    img = cv.imread('orange.jpg')  # 读取原始图像
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 转换为灰度图像
    kernelx = np.array([[-1, 0], [0, 1]], dtype=int)  # Roberts 算子的 x 方向核
    kernely = np.array([[0, -1], [1, 0]], dtype=int)  # Roberts 算子的 y 方向核

    x = cv.filter2D(gray, cv.CV_16S, kernelx)  # 应用 x 方向核进行滤波
    y = cv.filter2D(gray, cv.CV_16S, kernely)  # 应用 y 方向核进行滤波

    absX = cv.convertScaleAbs(x)  # 转换 x 方向的滤波结果为绝对值
    absY = cv.convertScaleAbs(y)  # 转换 y 方向的滤波结果为绝对值
    result = cv.addWeighted(absX, 0.5, absY, 0.5, 0)  # 合并 x 和 y 方向的结果，得到最终图像

    save_and_plot(img, result, 'roberts_sharpened', 'Roberts 算子')  # 保存和绘制对比图


# Sobel 算子锐化函数
def sobel_operator():
    img = cv.imread('orange.jpg')  # 读取原始图像
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 转换为灰度图像

    x = cv.Sobel(gray, cv.CV_16S, 1, 0)  # 应用 Sobel 算子进行 x 方向的滤波
    y = cv.Sobel(gray, cv.CV_16S, 0, 1)  # 应用 Sobel 算子进行 y 方向的滤波

    absX = cv.convertScaleAbs(x)  # 转换 x 方向的滤波结果为绝对值
    absY = cv.convertScaleAbs(y)  # 转换 y 方向的滤波结果为绝对值
    result = cv.addWeighted(absX, 0.5, absY, 0.5, 0)  # 合并 x 和 y 方向的结果，得到最终图像

    save_and_plot(img, result, 'sobel_sharpened', 'Sobel 算子')  # 保存和绘制对比图


# Prewitt 算子锐化函数
def prewitt_operator():
    img = cv.imread('orange.jpg')  # 读取原始图像
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 转换为灰度图像

    # Prewitt 算子的 x 和 y 方向核
    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)

    x = cv.filter2D(gray, cv.CV_16S, kernelx)  # 应用 x 方向核进行滤波
    y = cv.filter2D(gray, cv.CV_16S, kernely)  # 应用 y 方向核进行滤波

    absX = cv.convertScaleAbs(x)  # 转换 x 方向的滤波结果为绝对值
    absY = cv.convertScaleAbs(y)  # 转换 y 方向的滤波结果为绝对值
    result = cv.addWeighted(absX, 0.5, absY, 0.5, 0)  # 合并 x 和 y 方向的结果，得到最终图像

    save_and_plot(img, result, 'prewitt_sharpened', 'Prewitt 算子')  # 保存和绘制对比图


# Kirsch 算子锐化函数
def kirsch_operator():
    img = cv.imread('orange.jpg')  # 读取原始图像
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 转换为灰度图像

    # Kirsch 算子的多个方向的卷积核
    kirsch_kernels = [
        np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]], dtype=int),
        np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]], dtype=int),
        np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]], dtype=int),
        np.array([[-3, -3, -3], [5, 0, -3], [5, -3, -3]], dtype=int),
        np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]], dtype=int),
        np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]], dtype=int),
        np.array([[-3, -3, -3], [5, -3, -3], [5, -3, -3]], dtype=int),
        np.array([[-3, 5, 5], [-3, 0, -3], [-3, -3, -3]], dtype=int)
    ]

    # 对每个方向的核应用滤波并保存响应
    responses = [cv.filter2D(gray, cv.CV_16S, kernel) for kernel in kirsch_kernels]
    abs_responses = [cv.convertScaleAbs(response) for response in responses]  # 转换为绝对值
    result = np.max(abs_responses, axis=0)  # 取最大响应，得到最终图像

    save_and_plot(img, result, 'kirsch_sharpened', 'Kirsch 算子')  # 保存和绘制对比图


# Laplace 算子锐化函数
def laplace_sharpening():
    img = cv.imread('orange.jpg')  # 读取原始图像
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 转换为灰度图像

    laplacian = cv.Laplacian(gray, cv.CV_16S, ksize=3)  # 应用拉普拉斯算子进行滤波
    result = cv.convertScaleAbs(laplacian)  # 转换为绝对值，得到最终图像

    save_and_plot(img, result, 'laplace_sharpened', 'Laplace 二阶锐化')  # 保存和绘制对比图


# 主程序入口
num = input("请选择功能（1-Roberts, 2-Sobel, 3-Prewitt, 4-Kirsch, 5-Laplace）：")  # 提示用户选择功能

# 根据用户的输入调用相应的锐化函数
if num == '1':
    roberts()  # 调用 Roberts 算子锐化函数
elif num == '2':
    sobel_operator()  # 调用 Sobel 算子锐化函数
elif num == '3':
    prewitt_operator()  # 调用 Prewitt 算子锐化函数
elif num == '4':
    kirsch_operator()  # 调用 Kirsch 算子锐化函数
elif num == '5':
    laplace_sharpening()  # 调用 Laplace 算子锐化函数
else:
    print("无效输入，请输入1到5之间的数字。")  # 处理无效输入

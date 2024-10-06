import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']  # 选择字体
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题

#Roberts 算子
def roberts():
    # 读取图像并转换为灰度图
    img = cv.imread('orange.jpg')  # 读取图像，默认为彩色图像
    grayImage = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 转换为灰度图像

    # 定义 Roberts 算子
    kernelx = np.array([[-1, 0], [0, 1]], dtype=int)
    kernely = np.array([[0, -1], [1, 0]], dtype=int)

    # 应用 Roberts 算子
    x = cv.filter2D(grayImage, cv.CV_16S, kernelx)
    y = cv.filter2D(grayImage, cv.CV_16S, kernely)

    # 取绝对值并转换回 8 位图像
    absX = cv.convertScaleAbs(x)
    absY = cv.convertScaleAbs(y)

    # 计算 Roberts 边缘图像
    Roberts = cv.addWeighted(absX, 0.5, absY, 0.5, 0)

    # 显示原始图像和 Roberts 算子处理后的图像
    titles = ['原始图像', 'Roberts算子']
    images = [cv.cvtColor(img, cv.COLOR_BGR2RGB), Roberts]  # 转换原始图像为 RGB 以便 Matplotlib 显示

    for i in range(2):
        plt.subplot(1, 2, i + 1)
        plt.imshow(images[i], cmap='gray' if i == 1 else None)
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()

#sobel 算子
def sobel_operator():
    # 读取彩色图像
    img = cv.imread('orange.jpg')  # 读取彩色图像，默认为 BGR 格式
    rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # 转换为 RGB
    grayImage = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # 使用 Sobel 算子计算水平方向和垂直方向的梯度
    x = cv.Sobel(grayImage, cv.CV_16S, 1, 0)  # Sobel 水平方向梯度
    y = cv.Sobel(grayImage, cv.CV_16S, 0, 1)  # Sobel 垂直方向梯度

    # 取绝对值并转换为 8 位图像
    absX = cv.convertScaleAbs(x)
    absY = cv.convertScaleAbs(y)

    # 融合 x 和 y 方向的梯度，得到 Sobel 边缘检测结果
    Sobel = cv.addWeighted(absX, 0.5, absY, 0.5, 0)

    # 显示原始图像和 Sobel 算子处理后的图像
    titles = ['原始图像', 'Sobel 算子']
    images = [rgb_img, Sobel]

    # 创建子图来显示两张图像
    for i in range(2):
        plt.subplot(1, 2, i + 1)  # 创建一行两列的子图
        plt.imshow(images[i], cmap='gray' if i == 1 else None)  # 使用灰度显示处理图像
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])  # 隐藏刻度

    # 展示图像
    plt.show()

#Prewitt 算子
def prewitt_operator():
    # 读取图像为彩色图像
    img = cv.imread('orange.jpg')  # 读取彩色图像
    if img is None:
        print("图像读取失败，请检查图像路径和文件名是否正确。")
        return

    # 将图像转换为灰度图像
    grayImage = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 转换为灰度图像
    rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # 将原图转换为 RGB 格式以便 Matplotlib 显示

    # 定义 Prewitt 算子核（水平和垂直方向）
    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)  # 水平方向
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)  # 垂直方向

    # 使用 filter2D 进行卷积操作
    x = cv.filter2D(grayImage, cv.CV_16S, kernelx)  # 使用 Prewitt 算子的水平卷积
    y = cv.filter2D(grayImage, cv.CV_16S, kernely)  # 使用 Prewitt 算子的垂直卷积

    # 取绝对值并转换为 8 位图像
    absX = cv.convertScaleAbs(x)
    absY = cv.convertScaleAbs(y)

    # 合成 Prewitt 边缘检测结果
    Prewitt = cv.addWeighted(absX, 0.5, absY, 0.5, 0)

    # 用来正常显示中文标签
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 显示原始图像和 Prewitt 边缘检测结果
    titles = ['原始图像', 'Prewitt 算子']
    images = [rgb_img, Prewitt]

    for i in range(2):
        plt.subplot(1, 2, i + 1)  # 创建一行两列的子图
        plt.imshow(images[i], cmap='gray' if i == 1 else None)  # 使用灰度显示 Prewitt 结果
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])  # 隐藏坐标刻度
    plt.show()

#Kirsch算子
def kirsch_operator():
    # 读取彩色图像并转换为 RGB 格式
    img = cv.imread('orange.jpg')
    grayImage = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # 转换为 RGB 格式用于 Matplotlib 显示

    # 定义 Kirsch 算子8个方向的卷积核
    kirsch_kernels = [
        np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]], dtype=int),  # 上
        np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]], dtype=int),  # 右上
        np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]], dtype=int),  # 下
        np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]], dtype=int),  # 左上
        np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]], dtype=int),  # 右
        np.array([[-3, -3, -3], [-3, 0, -3], [-3, 5, 5]], dtype=int),  # 左
        np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, -3]], dtype=int),  # 左下
        np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]], dtype=int)  # 右下
    ]

    # 对每个方向使用 Kirsch 算子进行卷积运算
    kirsch_responses = [cv.filter2D(grayImage, cv.CV_16S, kernel) for kernel in kirsch_kernels]
    # 取各个方向卷积结果的绝对值
    abs_kirsch_responses = [cv.convertScaleAbs(response) for response in kirsch_responses]
    # 将各个方向的卷积结果取最大值，得到最终的 Kirsch 边缘检测结果
    kirsch_result = np.max(abs_kirsch_responses, axis=0)
    # 显示原始图像和 Kirsch 边缘检测结果
    titles = ['原始图像', 'Kirsch 算子']
    images = [rgb_img, kirsch_result]

    # 创建子图来显示两张图像
    for i in range(2):
        plt.subplot(1, 2, i + 1)  # 创建一行两列的子图
        plt.imshow(images[i], cmap='gray' if i == 1 else None)  # 使用灰度显示 Kirsch 结果
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])  # 隐藏坐标刻度

    plt.show()

#Laplace算子
def laplace_sharpening():
    # 读取图像（读取彩色图像）
    img = cv.imread('orange.jpg')
    grayImage = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # 转换为 RGB 格式以便 Matplotlib 显示

    # 使用 Laplacian 算子进行二阶锐化（默认使用 3x3 的卷积核）
    laplacian = cv.Laplacian(grayImage, cv.CV_16S, ksize=3)
    # 转换回 uint8 格式
    laplacian_abs = cv.convertScaleAbs(laplacian)

    # 显示原始图像和 Laplace 二阶锐化结果
    titles = ['原始图像', 'Laplace 二阶锐化']
    images = [rgb_img, laplacian_abs]

    # 创建子图来显示两张图像
    for i in range(2):
        plt.subplot(1, 2, i + 1)  # 创建一行两列的子图
        plt.imshow(images[i], cmap='gray' if i == 1 else None)  # 使用灰度显示锐化结果
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])  # 隐藏坐标刻度
    plt.show()


print("输入'1':对图像使用Roberts算子进行一阶锐化")
print("输入'2':对图像使用Roberts算子进行一阶锐化")
print("输入'3':对图像使用Prewitt算子进行一阶锐化")
print("输入'4':对图像使用kirsch算子进行一阶锐化")
print("输入'5':对图像使用Laplace算子进行二阶锐化")
num = input("请选择功能（使用图片orange.jpg）：")

if num == '1':
    roberts()
if num =='2':
    sobel_operator()
if num == '3':
    prewitt_operator()
if num == '4':
    kirsch_operator()
if num == '5':
    laplace_sharpening()

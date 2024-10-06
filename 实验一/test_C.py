import matplotlib.pyplot as plt  # 导入用于绘图的库
import cv2 as cv  # 导入OpenCV库，用于图像处理
import numpy as np  # 导入NumPy库，用于数值计算
import os  # 导入os库，用于文件和目录操作

# 设置绘图的字体和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为SimHei以支持中文显示
plt.rcParams['axes.unicode_minus'] = False  # 确保负号能正确显示

# 定义一个保存图像的函数
def save_image(image, name):
    # 创建保存目录，如果目录不存在则创建
    if not os.path.exists('sharpened_images'):
        os.makedirs('sharpened_images')  # 创建目录

    # 保存图像到指定路径
    cv.imwrite(f'sharpened_images/{name}.png', image)  # 使用OpenCV保存图像

# Roberts 算子锐化处理函数
def roberts():
    img = cv.imread('orange.jpg')  # 读取输入图像
    grayImage = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 将图像转换为灰度图

    # 定义Roberts算子的两个核
    kernelx = np.array([[-1, 0], [0, 1]], dtype=int)  # Roberts算子X方向的卷积核
    kernely = np.array([[0, -1], [1, 0]], dtype=int)  # Roberts算子Y方向的卷积核

    # 对图像应用Roberts算子
    x = cv.filter2D(grayImage, cv.CV_16S, kernelx)  # 使用X方向的核进行卷积
    y = cv.filter2D(grayImage, cv.CV_16S, kernely)  # 使用Y方向的核进行卷积

    absX = cv.convertScaleAbs(x)  # 将X方向结果转换为绝对值并缩放到8位图像
    absY = cv.convertScaleAbs(y)  # 将Y方向结果转换为绝对值并缩放到8位图像
    Roberts = cv.addWeighted(absX, 0.5, absY, 0.5, 0)  # 合并X和Y方向的结果

    # 保存锐化后的图像
    save_image(Roberts, 'roberts_sharpened')

    # 定义图像标题和内容
    titles = ['原始图像', 'Roberts算子']  # 图像标题
    images = [cv.cvtColor(img, cv.COLOR_BGR2RGB), Roberts]  # 原始图像与锐化图像

    # 绘制图像对比
    for i in range(2):
        plt.subplot(1, 2, i + 1)  # 创建1行2列的子图
        plt.imshow(images[i], cmap='gray' if i == 1 else None)  # 显示图像，若是锐化图像则为灰度
        plt.title(titles[i])  # 设置子图标题
        plt.xticks([]), plt.yticks([])  # 隐藏坐标轴刻度
    plt.show()  # 显示绘制的图像


# Sobel 算子锐化处理函数
def sobel_operator():
    img = cv.imread('orange.jpg')  # 读取输入图像
    rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # 将图像转换为RGB格式
    grayImage = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 将图像转换为灰度图

    # 对图像应用Sobel算子
    x = cv.Sobel(grayImage, cv.CV_16S, 1, 0)  # 计算X方向的梯度
    y = cv.Sobel(grayImage, cv.CV_16S, 0, 1)  # 计算Y方向的梯度

    absX = cv.convertScaleAbs(x)  # 将X方向结果转换为绝对值并缩放到8位图像
    absY = cv.convertScaleAbs(y)  # 将Y方向结果转换为绝对值并缩放到8位图像
    Sobel = cv.addWeighted(absX, 0.5, absY, 0.5, 0)  # 合并X和Y方向的结果

    # 保存锐化后的图像
    save_image(Sobel, 'sobel_sharpened')

    # 定义图像标题和内容
    titles = ['原始图像', 'Sobel 算子']  # 图像标题
    images = [rgb_img, Sobel]  # 原始图像与锐化图像

    # 绘制图像对比
    for i in range(2):
        plt.subplot(1, 2, i + 1)  # 创建1行2列的子图
        plt.imshow(images[i], cmap='gray' if i == 1 else None)  # 显示图像，若是锐化图像则为灰度
        plt.title(titles[i])  # 设置子图标题
        plt.xticks([]), plt.yticks([])  # 隐藏坐标轴刻度

    plt.show()  # 显示绘制的图像


# Prewitt 算子锐化处理函数
def prewitt_operator():
    img = cv.imread('orange.jpg')  # 读取输入图像
    if img is None:  # 检查图像是否读取成功
        print("图像读取失败，请检查图像路径和文件名是否正确。")  # 输出错误信息
        return  # 终止函数

    grayImage = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 将图像转换为灰度图
    rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # 将图像转换为RGB格式

    # 定义Prewitt算子的两个核
    kernelx = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]], dtype=int)  # Prewitt算子X方向的卷积核
    kernely = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=int)  # Prewitt算子Y方向的卷积核

    # 对图像应用Prewitt算子
    x = cv.filter2D(grayImage, cv.CV_16S, kernelx)  # 使用X方向的核进行卷积
    y = cv.filter2D(grayImage, cv.CV_16S, kernely)  # 使用Y方向的核进行卷积

    absX = cv.convertScaleAbs(x)  # 将X方向结果转换为绝对值并缩放到8位图像
    absY = cv.convertScaleAbs(y)  # 将Y方向结果转换为绝对值并缩放到8位图像
    Prewitt = cv.addWeighted(absX, 0.5, absY, 0.5, 0)  # 合并X和Y方向的结果

    # 保存锐化后的图像
    save_image(Prewitt, 'prewitt_sharpened')

    # 定义图像标题和内容
    titles = ['原始图像', 'Prewitt 算子']  # 图像标题
    images = [rgb_img, Prewitt]  # 原始图像与锐化图像

    # 绘制图像对比
    for i in range(2):
        plt.subplot(1, 2, i + 1)  # 创建1行2列的子图
        plt.imshow(images[i], cmap='gray' if i == 1 else None)  # 显示图像，若是锐化图像则为灰度
        plt.title(titles[i])  # 设置子图标题
        plt.xticks([]), plt.yticks([])  # 隐藏坐标轴刻度
    plt.show()  # 显示绘制的图像


# Kirsch 算子锐化处理函数
def kirsch_operator():
    img = cv.imread('orange.jpg')  # 读取输入图像
    grayImage = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 将图像转换为灰度图
    rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # 将图像转换为RGB格式

    # 定义Kirsch算子的8个方向的卷积核
    kirsch_kernels = [
        np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]], dtype=int),  # 北方向
        np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]], dtype=int),  # 东北方向
        np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]], dtype=int),  # 正南方向
        np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]], dtype=int),  # 西南方向
        np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]], dtype=int),  # 正东方向
        np.array([[-3, -3, -3], [-3, 0, -3], [-3, 5, 5]], dtype=int),  # 正西方向
        np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, -3]], dtype=int),  # 正北方向
        np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]], dtype=int)   # 西北方向
    ]

    # 对每个方向的核进行卷积并保存结果
    kirsch_responses = [cv.filter2D(grayImage, cv.CV_16S, kernel) for kernel in kirsch_kernels]  # 对每个核进行卷积
    abs_kirsch_responses = [cv.convertScaleAbs(response) for response in kirsch_responses]  # 转换为绝对值并缩放到8位图像
    kirsch_result = np.max(abs_kirsch_responses, axis=0)  # 获取最大响应结果

    # 保存锐化后的图像
    save_image(kirsch_result, 'kirsch_sharpened')

    # 定义图像标题和内容
    titles = ['原始图像', 'Kirsch 算子']  # 图像标题
    images = [rgb_img, kirsch_result]  # 原始图像与锐化图像

    # 绘制图像对比
    for i in range(2):
        plt.subplot(1, 2, i + 1)  # 创建1行2列的子图
        plt.imshow(images[i], cmap='gray' if i == 1 else None)  # 显示图像，若是锐化图像则为灰度
        plt.title(titles[i])  # 设置子图标题
        plt.xticks([]), plt.yticks([])  # 隐藏坐标轴刻度

    plt.show()  # 显示绘制的图像


# Laplace 算子锐化处理函数
def laplace_sharpening():
    img = cv.imread('orange.jpg')  # 读取输入图像
    grayImage = cv.cvtColor(img, cv.COLOR_BGR2GRAY)  # 将图像转换为灰度图
    rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)  # 将图像转换为RGB格式

    # 对图像应用Laplace算子
    laplacian = cv.Laplacian(grayImage, cv.CV_16S, ksize=3)  # 计算Laplace算子
    laplacian_abs = cv.convertScaleAbs(laplacian)  # 将结果转换为绝对值并缩放到8位图像

    # 保存锐化后的图像
    save_image(laplacian_abs, 'laplace_sharpened')

    # 定义图像标题和内容
    titles = ['原始图像', 'Laplace 二阶锐化']  # 图像标题
    images = [rgb_img, laplacian_abs]  # 原始图像与锐化图像

    # 绘制图像对比
    for i in range(2):
        plt.subplot(1, 2, i + 1)  # 创建1行2列的子图
        plt.imshow(images[i], cmap='gray' if i == 1 else None)  # 显示图像，若是锐化图像则为灰度
        plt.title(titles[i])  # 设置子图标题
        plt.xticks([]), plt.yticks([])  # 隐藏坐标轴刻度
    plt.show()  # 显示绘制的图像


# 主程序入口，提供用户选择
print("输入'1':对图像使用Roberts算子进行一阶锐化")  # 提示用户输入选项
print("输入'2':对图像使用Sobel算子进行一阶锐化")  # 提示用户输入选项
print("输入'3':对图像使用Prewitt算子进行一阶锐化")  # 提示用户输入选项
print("输入'4':对图像使用Kirsch算子进行一阶锐化")  # 提示用户输入选项
print("输入'5':对图像使用Laplace算子进行二阶锐化")  # 提示用户输入选项
num = input("请选择功能（使用图片orange.jpg）：")  # 获取用户输入

# 根据用户输入调用相应的函数
if num == '1':
    roberts()  # 调用Roberts算子函数
elif num == '2':
    sobel_operator()  # 调用Sobel算子函数
elif num == '3':
    prewitt_operator()  # 调用Prewitt算子函数
elif num == '4':
    kirsch_operator()  # 调用Kirsch算子函数
elif num == '5':
    laplace_sharpening()  # 调用Laplace算子函数
else:
    print("无效输入，请输入1到5之间的数字。")  # 提示用户输入无效

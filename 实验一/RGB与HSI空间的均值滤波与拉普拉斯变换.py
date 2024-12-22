import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置Matplotlib字体参数，支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体，支持中文
plt.rcParams['axes.unicode_minus'] = False  # 解决坐标轴负号显示问题


# 定义函数：在RGB空间上进行均值滤波和拉普拉斯变换
def process_rgb(img_rgb):
    # 对RGB图像进行均值滤波，卷积核大小为5x5
    img_blur = cv2.blur(img_rgb, (5, 5))

    # 对RGB图像进行拉普拉斯变换，计算图像边缘信息
    img_laplacian = cv2.Laplacian(img_rgb, cv2.CV_64F)

    # 将拉普拉斯变换结果取绝对值并转换为uint8类型
    img_laplacian = np.uint8(np.absolute(img_laplacian))

    return img_blur, img_laplacian  # 返回均值滤波后的图像和拉普拉斯变换后的图像


# 定义函数：将RGB图像转换为HSI图像
def rgb_to_hsi(img):
    img = img / 255.0  # 将RGB图像的值归一化到[0, 1]范围
    R = img[:, :, 0]  # 获取红色通道
    G = img[:, :, 1]  # 获取绿色通道
    B = img[:, :, 2]  # 获取蓝色通道

    # 计算亮度（I）：三个通道的平均值
    I = (R + G + B) / 3.0

    # 计算饱和度（S）：根据RGB通道的最小值与亮度进行计算
    min_val = np.minimum(np.minimum(R, G), B)  # 获取R、G、B中的最小值
    S = 1 - 3 * min_val / (R + G + B + 1e-6)  # 防止分母为0，故加上1e-6

    # 计算色调（H）：通过RGB的差值和一个复杂的三角计算来完成
    num = 0.5 * ((R - G) + (R - B))  # 色调计算的分子部分
    den = np.sqrt((R - G) ** 2 + (R - B) * (G - B)) + 1e-6  # 分母部分，加1e-6防止除以0
    theta = np.arccos(num / den)  # 计算角度（以弧度表示）

    H = np.zeros_like(R)  # 初始化色调矩阵
    H[B > G] = 2 * np.pi - theta[B > G]  # 根据条件调整色调角度
    H[B <= G] = theta[B <= G]  # 如果蓝色分量小于等于绿色，直接赋值

    H = H / (2 * np.pi)  # 将H值归一化到[0, 1]范围

    return H, S, I  # 返回色调（H）、饱和度（S）和亮度（I）


# 定义函数：将HSI图像转换回RGB
def hsi_to_rgb(H, S, I):
    H = H * 2 * np.pi  # 将色调值恢复到[0, 2π]范围

    # 初始化R、G、B通道的矩阵
    R, G, B = np.zeros_like(H), np.zeros_like(H), np.zeros_like(H)

    # 根据H值所在的区间，计算对应的RGB值
    # H在[0, 2π/3]，对应红色区域
    idx = (H >= 0) & (H < 2 * np.pi / 3)  # 找到H值在[0, 2π/3]范围内的像素位置
    B[idx] = I[idx] * (1 - S[idx])  # 计算蓝色通道
    R[idx] = I[idx] * (1 + (S[idx] * np.cos(H[idx])) / np.cos(np.pi / 3 - H[idx]))  # 计算红色通道
    G[idx] = 3 * I[idx] - (R[idx] + B[idx])  # 计算绿色通道

    # H在[2π/3, 4π/3]，对应绿色区域
    idx = (H >= 2 * np.pi / 3) & (H < 4 * np.pi / 3)  # 找到H值在[2π/3, 4π/3]范围内的像素位置
    H[idx] -= 2 * np.pi / 3  # 调整H值
    R[idx] = I[idx] * (1 - S[idx])  # 计算红色通道
    G[idx] = I[idx] * (1 + (S[idx] * np.cos(H[idx])) / np.cos(np.pi / 3 - H[idx]))  # 计算绿色通道
    B[idx] = 3 * I[idx] - (R[idx] + G[idx])  # 计算蓝色通道

    # H在[4π/3, 2π]，对应蓝色区域
    idx = (H >= 4 * np.pi / 3) & (H < 2 * np.pi)  # 找到H值在[4π/3, 2π]范围内的像素位置
    H[idx] -= 4 * np.pi / 3  # 调整H值
    G[idx] = I[idx] * (1 - S[idx])  # 计算绿色通道
    B[idx] = I[idx] * (1 + (S[idx] * np.cos(H[idx])) / np.cos(np.pi / 3 - H[idx]))  # 计算蓝色通道
    R[idx] = 3 * I[idx] - (G[idx] + B[idx])  # 计算红色通道

    # 将R、G、B合并为一个三通道图像，并将其值限制在[0, 255]范围内
    rgb = np.stack([R, G, B], axis=-1)
    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)  # 将浮点数转换为uint8类型

    return rgb  # 返回RGB图像


# 在HSI的强度（I）分量上进行均值滤波和拉普拉斯变换
def process_hsi(img_rgb):
    H, S, I = rgb_to_hsi(img_rgb)  # 将RGB图像转换为HSI

    # 对强度分量进行均值滤波
    I_blur = cv2.blur((I * 255).astype(np.uint8), (5, 5)) / 255.0

    # 对强度分量进行拉普拉斯变换
    I_laplacian = cv2.Laplacian((I * 255).astype(np.uint8), cv2.CV_64F)
    I_laplacian = np.uint8(np.absolute(I_laplacian)) / 255.0  # 拉普拉斯变换结果归一化

    # 将处理后的I分量与原始H、S合并
    img_hsi_blur = hsi_to_rgb(H, S, I_blur)  # 合并均值滤波后的I分量
    img_hsi_laplacian = hsi_to_rgb(H, S, I_laplacian)  # 合并拉普拉斯变换后的I分量

    return img_hsi_blur, img_hsi_laplacian  # 返回处理后的图像


# 读取彩色图像
img = cv2.imread('orange.jpg')  # 从文件中读取图像
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 将BGR图像转换为RGB

# 创建文件夹存储处理后的图像
output_dir = 'image_processing_results'  # 定义文件夹名称
if not os.path.exists(output_dir):  # 检查文件夹是否存在
    os.makedirs(output_dir)  # 不存在则创建

# 用户选择处理空间
print("请选择处理空间：")
print("1: RGB 空间均值滤波和拉普拉斯变换")
print("2: HSI 空间仅对强度分量均值滤波和拉普拉斯变换")
n = input("请输入选择(1 或 2): ")  # 获取用户输入

if n == '1':
    # 在RGB空间进行均值滤波和拉普拉斯变换
    img_blur, img_laplacian = process_rgb(img_rgb)

    # 绘制原始图像和处理后的图像
    plt.figure(figsize=(12, 6))  # 设置绘图窗口大小
    plt.subplot(1, 3, 1)  # 1行3列的第1个子图
    plt.imshow(img_rgb)  # 显示原始RGB图像
    plt.title('原始RGB图像')  # 设置标题
    plt.axis('off')  # 关闭坐标轴显示

    plt.subplot(1, 3, 2)  # 1行3列的第2个子图
    plt.imshow(img_blur)  # 显示均值滤波后的图像
    plt.title('RGB空间均值滤波')  # 设置标题
    plt.axis('off')  # 关闭坐标轴显示

    plt.subplot(1, 3, 3)  # 1行3列的第3个子图
    plt.imshow(img_laplacian)  # 显示拉普拉斯变换后的图像
    plt.title('RGB空间拉普拉斯变换')  # 设置标题
    plt.axis('off')  # 关闭坐标轴显示

    plt.tight_layout()  # 自动调整子图间的间距
    plt.savefig(f'{output_dir}/RGB_processing_results.png')  # 保存结果图像
    plt.show()  # 显示绘图窗口

if n == '2':
    # 在HSI空间进行处理
    img_hsi_blur, img_hsi_laplacian = process_hsi(img_rgb)

    # 绘制原始图像和处理后的图像
    plt.figure(figsize=(12, 6))  # 设置绘图窗口大小
    plt.subplot(1, 3, 1)  # 1行3列的第1个子图
    plt.imshow(img_rgb)  # 显示原始RGB图像
    plt.title('原始RGB图像')  # 设置标题
    plt.axis('off')  # 关闭坐标轴显示

    plt.subplot(1, 3, 2)  # 1行3列的第2个子图
    plt.imshow(img_hsi_blur)  # 显示均值滤波后的HSI图像
    plt.title('HSI空间均值滤波 (仅强度分量)')  # 设置标题
    plt.axis('off')  # 关闭坐标轴显示

    plt.subplot(1, 3, 3)  # 1行3列的第3个子图
    plt.imshow(img_hsi_laplacian)  # 显示拉普拉斯变换后的HSI图像
    plt.title('HSI空间拉普拉斯变换 (仅强度分量)')  # 设置标题
    plt.axis('off')  # 关闭坐标轴显示

    plt.tight_layout()  # 自动调整子图间的间距
    plt.savefig(f'{output_dir}/HSI_processing_results.png')  # 保存结果图像
    plt.show()  # 显示绘图窗口

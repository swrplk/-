import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置Matplotlib字体为SimHei以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
# 解决Matplotlib中坐标轴负号显示问题
plt.rcParams['axes.unicode_minus'] = False

# 创建一个用于保存图片的文件夹，如果不存在则创建
output_folder = 'RGB_HSI_images'
if not os.path.exists(output_folder):  # 如果文件夹不存在
    os.makedirs(output_folder)  # 创建文件夹


# 函数：保存并显示RGB分量图
def save_rgb_components(img_rgb, filename):
    # 分离图像的RGB通道
    R, G, B = cv2.split(img_rgb)
    # 创建三个图像矩阵分别存储R、G、B通道，其他通道置为0
    R_img = np.zeros_like(img_rgb)  # 创建与原图像大小相同的全零矩阵
    R_img[:, :, 0] = R  # 红色通道图像，其他通道为0

    G_img = np.zeros_like(img_rgb)  # 创建与原图像大小相同的全零矩阵
    G_img[:, :, 1] = G  # 绿色通道图像，其他通道为0

    B_img = np.zeros_like(img_rgb)  # 创建与原图像大小相同的全零矩阵
    B_img[:, :, 2] = B  # 蓝色通道图像，其他通道为0

    # 准备绘图，显示原图和各通道图像
    titles = ['原始彩色图像', '红色通道', '绿色通道', '蓝色通道']
    images = [img_rgb, R_img, G_img, B_img]

    plt.figure(figsize=(10, 8))  # 设置画布大小
    for i in range(4):  # 循环显示四个子图
        plt.subplot(2, 2, i + 1)  # 设置2x2的子图布局
        plt.imshow(images[i])  # 显示图像
        plt.title(titles[i])  # 设置图像标题
        plt.xticks([]), plt.yticks([])  # 隐藏坐标轴刻度

    plt.tight_layout()  # 自动调整子图布局避免重叠
    save_path = os.path.join(output_folder, filename)  # 设置保存路径
    plt.savefig(save_path)  # 保存图像到文件
    plt.show()  # 显示图像


# 函数：将RGB图像转换为HSI模型
def rgb_to_hsi(img):
    img = img / 255.0  # 将RGB值归一化到[0, 1]范围
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]  # 获取R、G、B分量

    # 计算亮度（Intensity）分量
    I = (R + G + B) / 3.0  # 亮度等于三个通道的平均值

    # 计算饱和度（Saturation）分量
    min_val = np.minimum(np.minimum(R, G), B)  # 找出RGB中最小值
    S = 1 - 3 * min_val / (R + G + B + 1e-6)  # 计算饱和度并避免除以零

    # 计算色调（Hue）分量
    num = 0.5 * ((R - G) + (R - B))  # 计算分子
    den = np.sqrt((R - G) ** 2 + (R - B) * (G - B)) + 1e-6  # 计算分母并避免除以零
    theta = np.arccos(num / den)  # 计算夹角（色调）

    H = np.zeros_like(R)  # 初始化色调矩阵
    H[B > G] = 2 * np.pi - theta[B > G]  # 如果B > G，色调为2π减去角度
    H[B <= G] = theta[B <= G]  # 否则直接为计算出的角度
    H = H / (2 * np.pi)  # 将色调归一化到[0, 1]范围

    return H, S, I  # 返回HSI三个分量


# 函数：保存并显示HSI分量图
def save_hsi_components(img_rgb, filename):
    H, S, I = rgb_to_hsi(img_rgb)  # 将RGB图像转换为HSI

    # 准备绘图，显示原图和HSI分量图
    titles = ['原始彩色图像', 'Hue (色调)', 'Saturation (饱和度)', 'Intensity (亮度)']
    images = [img_rgb, H, S, I]

    plt.figure(figsize=(10, 8))  # 设置画布大小
    for i in range(4):  # 循环显示四个子图
        plt.subplot(2, 2, i + 1)  # 设置2x2的子图布局
        if i == 0:
            plt.imshow(images[i])  # 显示原始彩色图像
        else:
            plt.imshow(images[i], cmap='gray')  # HSI分量用灰度图显示
        plt.title(titles[i])  # 设置图像标题
        plt.xticks([]), plt.yticks([])  # 隐藏坐标轴刻度

    plt.tight_layout()  # 自动调整子图布局避免重叠
    save_path = os.path.join(output_folder, filename)  # 设置保存路径
    plt.savefig(save_path)  # 保存图像到文件
    plt.show()  # 显示图像


# 读取图像文件并转换为RGB格式
img = cv2.imread('orange.jpg')  # 读取原始图像
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换为RGB格式

# 提示用户选择显示和保存RGB或HSI分量图
print("请选择要显示并保存的图像分量：")
print("1: RGB 分量图")
print("2: HSI 分量图")
n = input("请输入选择(1 或 2): ")  # 用户输入选择

if n == '1':
    save_rgb_components(img_rgb, 'RGB_components.png')  # 保存RGB分量图
    print("RGB 分量图已保存至 'RGB_HSI_images/RGB_components.png'")  # 提示用户
elif n == '2':
    save_hsi_components(img_rgb, 'HSI_components.png')  # 保存HSI分量图
    print("HSI 分量图已保存至 'RGB_HSI_images/HSI_components.png'")  # 提示用户
else:
    print("无效选择，请输入 1 或 2.")  # 提示无效输入

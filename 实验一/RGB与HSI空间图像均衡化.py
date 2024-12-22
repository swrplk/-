import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 设置字体以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 选择SimHei字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 创建文件夹以保存结果
output_dir = 'histogram_equalization_results'  # 指定输出文件夹名称
if not os.path.exists(output_dir):  # 检查文件夹是否存在
    os.makedirs(output_dir)  # 如果不存在，则创建该文件夹

# RGB空间直方图均衡化函数
def equalize_hist_rgb(img_rgb):
    # 分离RGB通道
    R, G, B = cv2.split(img_rgb)  # 将输入RGB图像分为三个通道

    # 对每个通道分别进行直方图均衡化
    R_eq = cv2.equalizeHist(R)  # 对红色通道进行均衡化
    G_eq = cv2.equalizeHist(G)  # 对绿色通道进行均衡化
    B_eq = cv2.equalizeHist(B)  # 对蓝色通道进行均衡化

    # 合并均衡化后的通道
    img_rgb_eq = cv2.merge((R_eq, G_eq, B_eq))  # 将均衡化后的三个通道合并为RGB图像

    return img_rgb_eq  # 返回均衡化后的RGB图像

# HSI空间转换函数（从RGB到HSI）
def rgb_to_hsi(img):
    img = img / 255.0  # 将图像数据归一化到[0, 1]范围
    R = img[:, :, 0]  # 提取红色通道
    G = img[:, :, 1]  # 提取绿色通道
    B = img[:, :, 2]  # 提取蓝色通道

    # 计算亮度（Intensity）
    I = (R + G + B) / 3.0  # 计算亮度分量

    # 计算饱和度（Saturation）
    min_val = np.minimum(np.minimum(R, G), B)  # 计算每个像素的最小值
    S = 1 - 3 * min_val / (R + G + B + 1e-6)  # 计算饱和度分量

    # 计算色调（Hue）
    num = 0.5 * ((R - G) + (R - B))  # 计算色调的分子
    den = np.sqrt((R - G) ** 2 + (R - B) * (G - B)) + 1e-6  # 计算色调的分母
    theta = np.arccos(num / den)  # 计算色调角度

    # 初始化Hue分量
    H = np.zeros_like(R)  # 创建与R相同大小的H数组
    H[B > G] = 2 * np.pi - theta[B > G]  # B通道大于G时的色调值
    H[B <= G] = theta[B <= G]  # B通道小于等于G时的色调值
    H = H / (2 * np.pi)  # 将H值归一化到[0, 1]

    return H, S, I  # 返回色调、饱和度和亮度分量

# HSI到RGB的转换函数
def hsi_to_rgb(H, S, I):
    H = H * 2 * np.pi  # 将H值恢复到[0, 2π]范围
    # 初始化RGB通道
    R, G, B = np.zeros_like(H), np.zeros_like(H), np.zeros_like(H)

    # H在[0, 2π/3]，对应红色区域
    idx = (H >= 0) & (H < 2 * np.pi / 3)  # 红色区域的索引
    B[idx] = I[idx] * (1 - S[idx])  # 计算蓝色通道
    R[idx] = I[idx] * (1 + (S[idx] * np.cos(H[idx])) / np.cos(np.pi / 3 - H[idx]))  # 计算红色通道
    G[idx] = 3 * I[idx] - (R[idx] + B[idx])  # 计算绿色通道

    # H在[2π/3, 4π/3]，对应绿色区域
    idx = (H >= 2 * np.pi / 3) & (H < 4 * np.pi / 3)  # 绿色区域的索引
    H[idx] -= 2 * np.pi / 3  # 色调值偏移
    R[idx] = I[idx] * (1 - S[idx])  # 计算红色通道
    G[idx] = I[idx] * (1 + (S[idx] * np.cos(H[idx])) / np.cos(np.pi / 3 - H[idx]))  # 计算绿色通道
    B[idx] = 3 * I[idx] - (R[idx] + G[idx])  # 计算蓝色通道

    # H在[4π/3, 2π]，对应蓝色区域
    idx = (H >= 4 * np.pi / 3) & (H < 2 * np.pi)  # 蓝色区域的索引
    H[idx] -= 4 * np.pi / 3  # 色调值偏移
    G[idx] = I[idx] * (1 - S[idx])  # 计算绿色通道
    B[idx] = I[idx] * (1 + (S[idx] * np.cos(H[idx])) / np.cos(np.pi / 3 - H[idx]))  # 计算蓝色通道
    R[idx] = 3 * I[idx] - (G[idx] + B[idx])  # 计算红色通道

    # 合并通道并转换回[0, 255]范围
    rgb = np.stack([R, G, B], axis=-1)  # 将R、G、B通道合并
    rgb = np.clip(rgb * 255, 0, 255).astype(np.uint8)  # 归一化并转换数据类型

    return rgb  # 返回RGB图像

# HSI空间直方图均衡化函数
def equalize_hist_hsi(img_rgb):
    H, S, I = rgb_to_hsi(img_rgb)  # 将RGB图像转换为HSI分量
    I_eq = cv2.equalizeHist((I * 255).astype(np.uint8)) / 255.0  # 对亮度通道进行直方图均衡化
    img_rgb_eq = hsi_to_rgb(H, S, I_eq)  # 将均衡化后的HSI转换回RGB图像
    return img_rgb_eq  # 返回均衡化后的RGB图像

# 读取彩色图像并转换为RGB格式
img = cv2.imread('orange.jpg')  # 读取图像
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 将BGR格式转换为RGB格式

# 执行RGB空间的直方图均衡化
img_rgb_eq = equalize_hist_rgb(img_rgb)  # 对RGB图像进行直方图均衡化
plt.figure(figsize=(10, 5))  # 创建一个10x5的图形窗口
plt.subplot(1, 2, 1)  # 创建子图1
plt.imshow(img_rgb)  # 显示原始RGB图像
plt.title('原始RGB图像')  # 设置子图标题
plt.xticks([]), plt.yticks([])  # 隐藏坐标轴刻度

plt.subplot(1, 2, 2)  # 创建子图2
plt.imshow(img_rgb_eq)  # 显示均衡化后的RGB图像
plt.title('RGB空间直方图均衡化')  # 设置子图标题
plt.xticks([]), plt.yticks([])  # 隐藏坐标轴刻度
plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域

# 保存RGB空间的直方图均衡化结果
plt.savefig(os.path.join(output_dir, 'rgb_hist_equalization.png'))  # 将当前图像保存到指定文件夹
plt.show()  # 显示图像

# 执行HSI空间的直方图均衡化
img_hsi_eq = equalize_hist_hsi(img_rgb)  # 对RGB图像进行HSI空间的直方图均衡化
plt.figure(figsize=(10, 5))  # 创建一个10x5的图形窗口
plt.subplot(1, 2, 1)  # 创建子图1
plt.imshow(img_rgb)  # 显示原始RGB图像
plt.title('原始RGB图像')  # 设置子图标题
plt.xticks([]), plt.yticks([])  # 隐藏坐标轴刻度

plt.subplot(1, 2, 2)  # 创建子图2
plt.imshow(img_hsi_eq)  # 显示均衡化后的HSI图像
plt.title('HSI空间直方图均衡化')  # 设置子图标题
plt.xticks([]), plt.yticks([])  # 隐藏坐标轴刻度
plt.tight_layout()  # 自动调整子图参数，使之填充整个图像区域

# 保存HSI空间的直方图均衡化结果
plt.savefig(os.path.join(output_dir, 'hsi_hist_equalization.png'))  # 将当前图像保存到指定文件夹
plt.show()  # 显示图像

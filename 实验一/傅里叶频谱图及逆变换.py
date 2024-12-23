import cv2
import numpy as np
import matplotlib.pyplot as plt
# 设置字体以支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def dft_and_idft(image_path):
    # 读取图像并转换为灰度图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # 将图像转换为浮点型
    img_float = np.float32(img)
    # 进行离散傅里叶变换
    dft = cv2.dft(img_float, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shifted = np.fft.fftshift(dft)  # 将低频移动到中心
    # 计算频谱
    magnitude_spectrum = cv2.magnitude(dft_shifted[:, :, 0], dft_shifted[:, :, 1])
    magnitude_spectrum = np.log(magnitude_spectrum + 1)  # 使用对数缩放
    # 进行逆离散傅里叶变换
    idft_shifted = np.fft.ifftshift(dft_shifted)  # 逆移位
    img_reconstructed = cv2.idft(idft_shifted)
    img_reconstructed = cv2.magnitude(img_reconstructed[:, :, 0], img_reconstructed[:, :, 1])  # 取幅值
    # 显示结果
    plt.figure(figsize=(12, 6))

    # 显示原图
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title('原始图片')
    plt.axis('off')

    # 显示频谱图
    plt.subplot(1, 3, 2)
    plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('频谱图')
    plt.axis('off')

    # 显示重建图像
    plt.subplot(1, 3, 3)
    plt.imshow(img_reconstructed, cmap='gray')
    plt.title('逆变换图')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# 调用函数，输入图像路径
dft_and_idft('orange.jpg')
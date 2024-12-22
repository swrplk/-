import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定黑体字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取灰度图像
image = cv2.imread('orange.jpg', cv2.IMREAD_GRAYSCALE)

# 添加高斯噪声的函数
def add_gaussian_noise(image, mean=0, sigma=25):
    noise = np.random.normal(mean, sigma, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return noisy_image

# 生成理想低通滤波器的函数
def ideal_lowpass_filter(shape, cutoff):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    mask = np.zeros((rows, cols), dtype=np.float32)
    y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
    mask[(x**2 + y**2) <= cutoff**2] = 1
    return mask

# 生成巴特沃斯低通滤波器的函数
def butterworth_lowpass_filter(shape, cutoff, order=2):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
    d = np.sqrt(x**2 + y**2)
    mask = 1 / (1 + (d / cutoff)**(2 * order))
    return mask

# 生成高斯低通滤波器的函数
def gaussian_lowpass_filter(shape, cutoff):
    rows, cols = shape
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[-crow:rows-crow, -ccol:cols-ccol]
    d = np.sqrt(x**2 + y**2)
    mask = np.exp(-(d**2) / (2 * (cutoff**2)))
    return mask

# 应用滤波器的函数
def apply_filter(image, mask):
    dft = np.fft.fft2(image)
    dft_shifted = np.fft.fftshift(dft)
    filtered = dft_shifted * mask
    f_ishifted = np.fft.ifftshift(filtered)
    img_back = np.fft.ifft2(f_ishifted)
    return np.abs(img_back)

# 添加噪声
noisy_image = add_gaussian_noise(image)

# 创建滤波器
cutoff = 30
ideal_mask = ideal_lowpass_filter(image.shape, cutoff)
butterworth_mask = butterworth_lowpass_filter(image.shape, cutoff)
gaussian_mask = gaussian_lowpass_filter(image.shape, cutoff)

# 应用滤波器
ideal_filtered = apply_filter(noisy_image, ideal_mask)
butterworth_filtered = apply_filter(noisy_image, butterworth_mask)
gaussian_filtered = apply_filter(noisy_image, gaussian_mask)

# 创建保存结果的目录
save_path = 'filtered_results'
if not os.path.exists(save_path):
    os.makedirs(save_path)

# 合并结果图为一张大图
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

axs[0, 0].imshow(image, cmap='gray')
axs[0, 0].set_title('原始图像')
axs[0, 0].axis('off')

axs[0, 1].imshow(noisy_image, cmap='gray')
axs[0, 1].set_title('含噪声图像')
axs[0, 1].axis('off')

axs[0, 2].imshow(ideal_filtered, cmap='gray')
axs[0, 2].set_title('理想低通滤波结果')
axs[0, 2].axis('off')

axs[1, 0].imshow(butterworth_filtered, cmap='gray')
axs[1, 0].set_title('巴特沃斯低通滤波结果')
axs[1, 0].axis('off')

axs[1, 1].imshow(gaussian_filtered, cmap='gray')
axs[1, 1].set_title('高斯低通滤波结果')
axs[1, 1].axis('off')

# 留空的子图
axs[1, 2].axis('off')

# 保存合并图
plt.tight_layout()
plt.savefig(os.path.join(save_path, 'filtered_results.png'))
plt.show()

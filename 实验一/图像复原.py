import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from skimage import util, restoration

# 创建保存图像的目录
output_dir = '图像复原'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 读取图像并转换为灰度图像
img_color = cv2.imread('orange.jpg')
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

# 显示图像及其直方图的函数，并保存
def show_image_and_histogram(image, title, save_name):
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.hist(image.ravel(), bins=256, range=[0, 256], density=True)
    plt.title(f"{title} Histogram")

    # 保存图像
    plt.savefig(os.path.join(output_dir, f"{save_name}.png"))
    plt.close()


# 1. 添加高斯噪声
def add_gaussian_noise(image, mean=0, sigma=25):
    gauss_noise = np.random.normal(mean, sigma, image.shape).astype(np.uint8)
    noisy_img = cv2.add(image, gauss_noise)
    return noisy_img


# 2. 添加均匀噪声
def add_uniform_noise(image, low=-50, high=50):
    uniform_noise = np.random.uniform(low, high, image.shape).astype(np.uint8)
    noisy_img = cv2.add(image, uniform_noise)
    return noisy_img


# 3. 添加椒盐噪声
def add_salt_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    noisy_img = util.random_noise(image, mode='s&p', amount=salt_prob + pepper_prob)
    noisy_img = (255 * noisy_img).astype(np.uint8)
    return noisy_img


# 显示原图并保存
show_image_and_histogram(img_gray, "Original Gray Image", "original_gray_image")

# 显示高斯噪声图像及直方图并保存
img_gaussian_noise = add_gaussian_noise(img_gray)
show_image_and_histogram(img_gaussian_noise, "Gaussian Noise Image", "gaussian_noise_image")

# 显示均匀噪声图像及直方图并保存
img_uniform_noise = add_uniform_noise(img_gray)
show_image_and_histogram(img_uniform_noise, "Uniform Noise Image", "uniform_noise_image")

# 显示椒盐噪声图像及直方图并保存
img_salt_pepper_noise = add_salt_pepper_noise(img_gray)
show_image_and_histogram(img_salt_pepper_noise, "Salt & Pepper Noise Image", "salt_pepper_noise_image")

# 使用滤波器进行降噪
# 高斯噪声 -> 高斯滤波
img_gaussian_denoised = cv2.GaussianBlur(img_gaussian_noise, (5, 5), 1)

# 均匀噪声 -> 中值滤波
img_uniform_denoised = cv2.medianBlur(img_uniform_noise, 5)

# 椒盐噪声 -> 中值滤波
img_salt_pepper_denoised = cv2.medianBlur(img_salt_pepper_noise, 5)


# 显示降噪前后对比并保存
def show_denoise_comparison(noisy_img, denoised_img, title, save_name):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(noisy_img, cmap='gray')
    plt.title(f"Noisy {title}")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(denoised_img, cmap='gray')
    plt.title(f"Denoised {title}")
    plt.axis('off')

    # 保存对比图
    plt.savefig(os.path.join(output_dir, f"{save_name}_denoise_comparison.png"))
    plt.close()


# 显示高斯噪声去噪对比并保存
show_denoise_comparison(img_gaussian_noise, img_gaussian_denoised, "Gaussian Noise", "gaussian_noise")

# 显示均匀噪声去噪对比并保存
show_denoise_comparison(img_uniform_noise, img_uniform_denoised, "Uniform Noise", "uniform_noise")

# 显示椒盐噪声去噪对比并保存
show_denoise_comparison(img_salt_pepper_noise, img_salt_pepper_denoised, "Salt & Pepper Noise", "salt_pepper_noise")

# 运动模糊
def motion_blur(image, kernel_size=15):
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
    kernel = kernel / kernel_size
    motion_blurred = convolve2d(image, kernel, mode='same', boundary='wrap')
    return motion_blurred


# 添加运动模糊和高斯噪声
img_motion_blur = motion_blur(img_gray)
img_motion_blur_gaussian = add_gaussian_noise(img_motion_blur.astype(np.uint8))

# 维纳滤波恢复
img_wiener_restored = restoration.wiener(img_motion_blur_gaussian, np.ones((5, 5)) / 25, 1)

# 约束最小二乘滤波恢复
psf = np.ones((15, 15)) / 15  # Point Spread Function
img_constrained_restored = restoration.unsupervised_wiener(img_motion_blur_gaussian, psf)[0]

# 显示运动模糊加噪图像与恢复后的对比并保存
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(img_motion_blur_gaussian, cmap='gray')
plt.title("Motion Blurred + Gaussian Noise")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(img_wiener_restored, cmap='gray')
plt.title("Wiener Filter Restored")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(img_constrained_restored, cmap='gray')
plt.title("Constrained Least Squares Restored")
plt.axis('off')

# 保存对比图
plt.savefig(os.path.join(output_dir, "motion_blur_restoration_comparison.png"))
plt.close()

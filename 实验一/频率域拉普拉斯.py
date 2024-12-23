import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 设置支持中文的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定黑体字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取灰度图像
image = cv2.imread('orange.jpg', cv2.IMREAD_GRAYSCALE)


# 计算拉普拉斯算子的函数
def laplacian_filter(image):
    # 获取图像的 DFT
    dft = np.fft.fft2(image)
    dft_shifted = np.fft.fftshift(dft)

    # 创建拉普拉斯滤波器
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[-crow:rows - crow, -ccol:cols - ccol]
    laplacian_mask = -4 * (np.pi ** 2) * (x ** 2 + y ** 2)

    # 应用拉普拉斯滤波器
    filtered_dft = dft_shifted * laplacian_mask
    f_ishifted = np.fft.ifftshift(filtered_dft)
    img_back = np.fft.ifft2(f_ishifted)

    # 取绝对值并进行归一化
    enhanced_image = np.abs(img_back)
    enhanced_image = cv2.normalize(enhanced_image, None, 0, 255, cv2.NORM_MINMAX)
    return enhanced_image.astype(np.uint8)


# 应用拉普拉斯滤波
enhanced_image = laplacian_filter(image)

# 创建保存结果的目录
output_dir = 'results'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 合并结果图
combined_image = np.hstack((image, enhanced_image))

# 保存合并的结果图
output_path = os.path.join(output_dir, 'combined_result.png')
cv2.imwrite(output_path, combined_image)

# 可视化结果
plt.figure(figsize=(12, 6))
plt.imshow(combined_image, cmap='gray')
plt.title('原始图像与拉普拉斯算子增强图像对比')
plt.axis('off')
plt.tight_layout()
plt.show()

print(f"合并结果图已保存至: {output_path}")

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 创建结果保存目录
result_dir = "harris_corner_results"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# 读取图像
image_path = "orange.jpg"
image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转为灰度图

# Harris 角点检测
gray_image = np.float32(gray_image)  # 转换为 float32 类型
dst = cv2.cornerHarris(gray_image, blockSize=2, ksize=3, k=0.04)  # Harris 检测
dst = cv2.dilate(dst, None)  # 膨胀角点，提高可视化效果

# 标记角点
corner_image = image.copy()
corner_image[dst > 0.01 * dst.max()] = [0, 0, 255]  # 角点标红（阈值为 0.01）

# 保存结果
original_path = os.path.join(result_dir, "original_image.jpg")
gray_path = os.path.join(result_dir, "gray_image.jpg")
corner_path = os.path.join(result_dir, "harris_corners.jpg")

cv2.imwrite(original_path, image)
cv2.imwrite(gray_path, cv2.cvtColor(gray_image.astype(np.uint8), cv2.COLOR_GRAY2BGR))
cv2.imwrite(corner_path, corner_image)

# 可视化结果
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(gray_image, cmap='gray')
plt.title("Gray Image")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(corner_image, cv2.COLOR_BGR2RGB))
plt.title("Harris Corners")
plt.axis("off")

# 保存可视化结果
visualization_path = os.path.join(result_dir, "harris_corners_visualization.jpg")
plt.savefig(visualization_path)
plt.show()

print(f"Harris 角点检测完成，结果保存在目录: {result_dir}")

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 创建结果保存目录
result_dir = "hog_image_results"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# 读取图像
image_path = "orange.jpg"
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 转为灰度图
cv2.imwrite(os.path.join(result_dir, "gray_image.jpg"), image)

# 检查图像大小并调整大小以便于计算
image = cv2.resize(image, (128, 128))  # 将图像调整为128x128的标准大小

# 初始化 HOG 描述器
hog = cv2.HOGDescriptor()

# 计算 HOG 特征
hog_features = hog.compute(image)

# 归一化 HOG 特征
hog_features_normalized = hog_features / np.linalg.norm(hog_features)

# 绘制归一化后的 HOG 直方图
plt.figure(figsize=(10, 5))
plt.plot(hog_features_normalized, color='blue')
plt.title("Normalized HOG Features Histogram")
plt.xlabel("Feature Index")
plt.ylabel("Normalized Value")
plt.grid()
plt.tight_layout()

# 保存直方图
histogram_path = os.path.join(result_dir, "hog_histogram.jpg")
plt.savefig(histogram_path)

plt.show()

print(f"HOG特征提取完成，灰度图和直方图结果保存到目录: {result_dir}")

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 创建保存结果的目录
result_dir = "pca_image_results"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# 读取图像
image_path = "orange.jpg"
image = cv2.imread(image_path, cv2.IMREAD_COLOR)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为RGB格式
cv2.imwrite(os.path.join(result_dir, "original_image.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

# 图像数据预处理
height, width, channels = image.shape
restored_channels = []

# 对每个通道单独进行 PCA
for channel in range(channels):
    single_channel = image[:, :, channel]  # 提取单通道
    reshaped_channel = single_channel.reshape(height, -1)  # 展平为二维 (height, width)

    # 主成分分析 (PCA)
    pca = PCA(n_components=min(height, width) // 2)  # 设置合理的主成分数量
    transformed = pca.fit_transform(reshaped_channel)
    restored = pca.inverse_transform(transformed)  # 恢复通道

    restored_channels.append(restored.astype(np.uint8))

# 合并恢复后的通道
restored_image = np.stack(restored_channels, axis=2)

# 保存恢复后的图像
restored_image_path = os.path.join(result_dir, "restored_image.jpg")
cv2.imwrite(restored_image_path, cv2.cvtColor(restored_image, cv2.COLOR_RGB2BGR))

# 可视化原图和恢复图
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(restored_image)
plt.title("Restored Image")
plt.axis("off")

plt.tight_layout()
plt.savefig(os.path.join(result_dir, "comparison_plot.jpg"))
plt.show()

print("图像处理完成，结果保存在:", result_dir)

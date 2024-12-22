import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 创建结果保存目录
result_dir = "hough_transform_results"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# 读取图像并转换为灰度图
image_path = "orange.jpg"
image = cv2.imread(image_path)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转为灰度图
cv2.imwrite(os.path.join(result_dir, "gray_image.jpg"), gray_image)

# 边缘检测
edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)  # Canny 边缘检测
cv2.imwrite(os.path.join(result_dir, "edges.jpg"), edges)

# 直线检测（霍夫变换）
lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)  # 霍夫直线变换
line_image = image.copy()

if lines is not None:
    for rho, theta in lines[:, 0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imwrite(os.path.join(result_dir, "hough_lines.jpg"), line_image)

# 圆检测（霍夫圆变换）
circles = cv2.HoughCircles(
    gray_image,
    cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=20,
    param1=50,
    param2=30,
    minRadius=5,
    maxRadius=50
)
circle_image = image.copy()

if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        center = (i[0], i[1])  # 圆心
        radius = i[2]  # 半径
        cv2.circle(circle_image, center, radius, (255, 0, 0), 2)  # 绘制圆
        cv2.circle(circle_image, center, 2, (0, 255, 0), 3)  # 绘制圆心

cv2.imwrite(os.path.join(result_dir, "hough_circles.jpg"), circle_image)

# 可视化结果
plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(edges, cmap='gray')
plt.title("Edges")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.imshow(cv2.cvtColor(line_image, cv2.COLOR_BGR2RGB))
plt.title("Hough Lines")
plt.axis("off")

plt.subplot(2, 2, 4)
plt.imshow(cv2.cvtColor(circle_image, cv2.COLOR_BGR2RGB))
plt.title("Hough Circles")
plt.axis("off")

# 保存可视化结果
visualization_path = os.path.join(result_dir, "hough_transform_visualization.jpg")
plt.savefig(visualization_path)
plt.show()

print(f"霍夫变换检测完成，结果保存在目录: {result_dir}")

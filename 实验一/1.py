import cv2
import numpy as np
import matplotlib.pyplot as plt

# 加载彩色图像并灰度化
image = cv2.imread('orange.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# Prewitt梯度算子
def prewitt_edge_detection(img):
    # Prewitt算子核
    kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])

    # 卷积操作
    grad_x = cv2.filter2D(img, -1, kernel_x)
    grad_y = cv2.filter2D(img, -1, kernel_y)

    # 梯度幅值
    magnitude = cv2.magnitude(grad_x.astype(float), grad_y.astype(float))
    return grad_x, grad_y, magnitude


# 应用Prewitt梯度算子
grad_x, grad_y, magnitude = prewitt_edge_detection(gray)

# 显示结果
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1), plt.title('Gradient X'), plt.imshow(grad_x, cmap='gray')
plt.subplot(1, 3, 2), plt.title('Gradient Y'), plt.imshow(grad_y, cmap='gray')
plt.subplot(1, 3, 3), plt.title('Magnitude'), plt.imshow(magnitude, cmap='gray')
plt.show()

# 平滑处理（高斯滤波）
smoothed = cv2.GaussianBlur(gray, (5, 5), 1.0)
grad_x_smooth, grad_y_smooth, magnitude_smooth = prewitt_edge_detection(smoothed)

# 显示平滑后的结果
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1), plt.title('Smoothed Gradient X'), plt.imshow(grad_x_smooth, cmap='gray')
plt.subplot(1, 3, 2), plt.title('Smoothed Gradient Y'), plt.imshow(grad_y_smooth, cmap='gray')
plt.subplot(1, 3, 3), plt.title('Smoothed Magnitude'), plt.imshow(magnitude_smooth, cmap='gray')
plt.show()


# 对角线Prewitt梯度算子
def diagonal_prewitt(img):
    # 对角线Prewitt算子核
    kernel_45 = np.array([[0, 1, 1], [-1, 0, 1], [-1, -1, 0]])
    kernel_135 = np.array([[1, 1, 0], [1, 0, -1], [0, -1, -1]])

    # 卷积操作
    grad_45 = cv2.filter2D(img, -1, kernel_45)
    grad_135 = cv2.filter2D(img, -1, kernel_135)

    # 梯度幅值
    magnitude_diag = cv2.magnitude(grad_45.astype(float), grad_135.astype(float))
    return grad_45, grad_135, magnitude_diag


grad_45, grad_135, magnitude_diag = diagonal_prewitt(gray)

# 显示对角线Prewitt结果
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1), plt.title('Gradient 45°'), plt.imshow(grad_45, cmap='gray')
plt.subplot(1, 3, 2), plt.title('Gradient 135°'), plt.imshow(grad_135, cmap='gray')
plt.subplot(1, 3, 3), plt.title('Diagonal Magnitude'), plt.imshow(magnitude_diag, cmap='gray')
plt.show()

# 阈值化处理
_, binary_magnitude = cv2.threshold(magnitude, 50, 255, cv2.THRESH_BINARY)
_, binary_magnitude_diag = cv2.threshold(magnitude_diag, 50, 255, cv2.THRESH_BINARY)

# 显示阈值化结果
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1), plt.title('Thresholded Magnitude'), plt.imshow(binary_magnitude, cmap='gray')
plt.subplot(1, 2, 2), plt.title('Thresholded Diagonal Magnitude'), plt.imshow(binary_magnitude_diag, cmap='gray')
plt.show()
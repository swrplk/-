import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 创建保存结果的文件夹
output_dir = '图像分割'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 加载彩色图像并灰度化
image = cv2.imread('orange.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Prewitt梯度算子边缘检测
def prewitt_edge_detection(img):
    # 定义Prewitt算子的两个方向的卷积核
    kernel_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])  # 水平方向
    kernel_y = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])   # 垂直方向

    # 使用filter2D函数进行卷积操作，分别计算水平和垂直方向的梯度
    grad_x = cv2.filter2D(img, -1, kernel_x)
    grad_y = cv2.filter2D(img, -1, kernel_y)

    # 计算梯度幅值，表示边缘的强度
    magnitude = cv2.magnitude(grad_x.astype(float), grad_y.astype(float))
    return grad_x, grad_y, magnitude

# 应用Prewitt算子进行边缘检测
grad_x, grad_y, magnitude = prewitt_edge_detection(gray)

# 显示Prewitt算子结果（梯度方向及幅值）
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1), plt.title('Gradient X'), plt.imshow(grad_x, cmap='gray')
plt.subplot(1, 3, 2), plt.title('Gradient Y'), plt.imshow(grad_y, cmap='gray')
plt.subplot(1, 3, 3), plt.title('Magnitude'), plt.imshow(magnitude, cmap='gray')
plt.savefig(os.path.join(output_dir, 'prewitt_edges.png'))
plt.close()

# 对图像进行高斯平滑，去除噪声
smoothed = cv2.GaussianBlur(gray, (5, 5), 1.0)
grad_x_smooth, grad_y_smooth, magnitude_smooth = prewitt_edge_detection(smoothed)

# 显示平滑处理后的结果
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1), plt.title('Smoothed Gradient X'), plt.imshow(grad_x_smooth, cmap='gray')
plt.subplot(1, 3, 2), plt.title('Smoothed Gradient Y'), plt.imshow(grad_y_smooth, cmap='gray')
plt.subplot(1, 3, 3), plt.title('Smoothed Magnitude'), plt.imshow(magnitude_smooth, cmap='gray')
plt.savefig(os.path.join(output_dir, 'prewitt_edges_smoothed.png'))
plt.close()

# 对角线Prewitt梯度算子（包括45°和135°方向）
def diagonal_prewitt(img):
    # 定义对角线Prewitt算子的两个方向的卷积核：45°和135°方向
    kernel_45 = np.array([[0, 1, 1], [-1, 0, 1], [-1, -1, 0]])   # 45°方向
    kernel_135 = np.array([[1, 1, 0], [1, 0, -1], [0, -1, -1]])  # 135°方向

    # 使用filter2D函数进行卷积操作，计算两个方向的梯度
    grad_45 = cv2.filter2D(img, -1, kernel_45)
    grad_135 = cv2.filter2D(img, -1, kernel_135)

    # 计算对角线梯度幅值
    magnitude_diag = cv2.magnitude(grad_45.astype(float), grad_135.astype(float))
    return grad_45, grad_135, magnitude_diag

# 应用对角线Prewitt算子进行边缘检测
grad_45, grad_135, magnitude_diag = diagonal_prewitt(gray)

# 显示对角线Prewitt算子结果（45°和135°方向的梯度及幅值）
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1), plt.title('Gradient 45°'), plt.imshow(grad_45, cmap='gray')
plt.subplot(1, 3, 2), plt.title('Gradient 135°'), plt.imshow(grad_135, cmap='gray')
plt.subplot(1, 3, 3), plt.title('Diagonal Magnitude'), plt.imshow(magnitude_diag, cmap='gray')
plt.savefig(os.path.join(output_dir, 'diagonal_prewitt_edges.png'))
plt.close()

# 使用Canny算子进行边缘检测
edges_canny = cv2.Canny(gray, 100, 200)  # 低阈值100，高阈值200

# 显示Canny算子的检测结果
plt.figure(figsize=(6, 6))
plt.title('Canny Edge Detection'), plt.imshow(edges_canny, cmap='gray')
plt.savefig(os.path.join(output_dir, 'canny_edges.png'))
plt.close()

# Otsu方法直接进行图像分割
# 使用Otsu阈值化方法
_, otsu_thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 显示Otsu分割结果
plt.figure(figsize=(6, 6))
plt.title('Otsu Thresholding'), plt.imshow(otsu_thresholded, cmap='gray')
plt.savefig(os.path.join(output_dir, 'otsu_thresholding.png'))
plt.close()

# 边缘图像的阈值化处理
# 使用Prewitt边缘检测的梯度幅值进行阈值化
_, binary_prewitt = cv2.threshold(magnitude, 50, 255, cv2.THRESH_BINARY)
# 使用Otsu方法处理分割后的图像
_, binary_otsu = cv2.threshold(otsu_thresholded, 127, 255, cv2.THRESH_BINARY)

# 显示二值化结果进行比较
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1), plt.title('Prewitt Thresholded'), plt.imshow(binary_prewitt, cmap='gray')
plt.subplot(1, 2, 2), plt.title('Otsu Thresholded'), plt.imshow(binary_otsu, cmap='gray')
plt.savefig(os.path.join(output_dir, 'threshold_comparison.png'))
plt.close()

# 阈值化后的幅值图像
_, binary_magnitude = cv2.threshold(magnitude, 50, 255, cv2.THRESH_BINARY)
_, binary_magnitude_diag = cv2.threshold(magnitude_diag, 50, 255, cv2.THRESH_BINARY)

# 保存阈值化后的幅值图像
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1), plt.title('Thresholded Magnitude'), plt.imshow(binary_magnitude, cmap='gray')
plt.subplot(1, 2, 2), plt.title('Thresholded Diagonal Magnitude'), plt.imshow(binary_magnitude_diag, cmap='gray')
plt.savefig(os.path.join(output_dir, 'thresholded_magnitude.png'))
plt.close()

print("所有结果图已保存至文件夹 '图像分割'")

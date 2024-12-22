import matplotlib.pyplot as plt
import cv2
import numpy as np

# 设置字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 选择字体
plt.rcParams['axes.unicode_minus'] = False   # 解决负号显示问题

# 手动添加高斯噪声的函数
def gasuss_noise(image, mean=0, var=0.001):
    '''
    手动添加高斯噪声
    mean : 均值
    var : 方差
    '''
    image = np.array(image / 255, dtype=float)
    noise = np.random.normal(mean, var ** 0.5, image.shape)  # 正态分布
    out = image + noise
    out = np.clip(out, 0, 1.0)  # 限制值的范围在0-1之间
    out = np.uint8(out * 255)
    return out

# 读取图像并生成高斯噪声图像
img = cv2.imread('orange.jpg')
noisy_img = gasuss_noise(img)

# 将带噪声的图像保存为文件
cv2.imwrite('my_gasuss_noise.jpg', noisy_img)

# 读取带噪声的图像
img_noise = cv2.imread('my_gasuss_noise.jpg')
# 图像平滑处理
#均值滤波
img_mean = cv2.blur(img_noise, (5, 5))
#方框滤波
img_box = cv2.boxFilter(img_noise, -1, (5, 5), normalize=1)
#高斯滤波
img_gaussian = cv2.GaussianBlur(img_noise, ksize=(3,3), sigmaX=1.2, sigmaY=0.8)
# 显示图像对比
titles = ['原始图像', '带噪声图像', '均值滤波', '方框滤波', '高斯滤波']
images = [img, img_noise, img_mean, img_box, img_gaussian]
plt.figure(figsize=(10, 8))
for i in range(5):
    plt.subplot(2, 3, i + 1)
    plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))  # 转换为RGB显示
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.show()
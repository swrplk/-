import matplotlib.pyplot as plt
import cv2
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei']  # 选择字体为SimHei（黑体）
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

#把灰度图进行切片
def hui_du_ji_qie_pian(img):  # 定义灰度图切片函数
    h, w = img.shape[0], img.shape[1]  # 获取图像的高度和宽度
    new_img = np.zeros((h, w))  # 创建一个与原图同样大小的全零数组
    for i in range(h):  # 遍历每一行
        for j in range(w):  # 遍历每一列
            # 灰度小于230大于190就设置为255
            if img[i, j] <= 230 and img[i, j] >= 190:  # 检查灰度值
                new_img[i, j] = 255  # 设置为255（白色）
            else:
                new_img[i, j] = img[i, j]  # 保持原灰度值

    return new_img  # 返回切片后的新图像

def hua_T_r():  # 定义灰度级切片转换函数
    r = np.arange(0, 256)  # 创建从0到255的灰度值数组
    # 按照条件设置输出灰度值
    T = np.where((r >= 190) & (r <= 230), 255, r)  # 在190到230范围内输出255，其余保持不变

    # 绘制转换函数图像
    plt.plot(r, T, color='blue')  # 绘制转换函数
    plt.title('T(r) - 灰度级切片转换函数')  # 设置图标题
    plt.xlabel('输入灰度值 r')  # 设置x轴标签
    plt.ylabel('输出灰度值 T(r)')  # 设置y轴标签
    plt.grid(True)  # 显示网格
    plt.show()  # 展示图像

# 位平面切片
def wei_ping_mian_qie_pian():  # 定义位平面切片函数
    img = cv2.imread('gray_image.png', cv2.IMREAD_GRAYSCALE)  # 读取灰度图像
    bit_planes = []  # 创建一个列表存储8个位平面的图像

    for i in range(8):  # 遍历每一位平面
        bit_plane = (img >> i) & 1  # 使用位移操作提取每个位平面
        bit_planes.append(bit_plane * 255)  # 将二值图像放大到0-255范围

    # 创建一个图像网格展示所有位平面
    plt.figure(figsize=(10, 6))  # 创建绘图窗口
    for i in range(8):  # 遍历每个位平面
        plt.subplot(2, 4, i + 1)  # 创建子图
        plt.imshow(bit_planes[i], cmap='gray')  # 显示当前位平面
        plt.title(f'Bit Plane {i}')  # 设置子图标题
        plt.axis('off')  # 关闭坐标轴

    plt.tight_layout()  # 自动调整子图布局
    plt.show()  # 展示图像

#算图像的直方图
def calc_histogram_image(image):  # 定义计算直方图的函数
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])  # 计算灰度直方图
    fig, ax = plt.subplots()  # 创建图形和坐标轴
    ax.plot(hist, color='black')  # 绘制直方图
    ax.set_xlim([0, 256])  # 设置x轴范围
    ax.set_title('Histogram')  # 设置标题

    # 使用 buffer_rgba 替代 tostring_rgb
    fig.canvas.draw()  # 绘制当前图形
    histogram_img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)  # 从缓冲区获取RGBA数据

    # 将RGBA图像转为RGB格式
    histogram_img = histogram_img.reshape(fig.canvas.get_width_height()[::-1] + (4,))  # 重塑为图像
    histogram_img = histogram_img[:, :, :3]  # 只取RGB通道，丢弃Alpha通道
    plt.close(fig)  # 关闭figure，释放内存
    return histogram_img  # 返回直方图图像

# 创建暗图像
def create_dark_image(img):  # 定义创建暗图像的函数
    dark_img = np.clip(img * 0.5, 0, 255).astype(np.uint8)  # 亮度减半，限制在0到255之间
    return dark_img  # 返回暗图像

# 创建亮图像
def create_bright_image(img):  # 定义创建亮图像的函数
    bright_img = np.clip(img * 1.5, 0, 255).astype(np.uint8)  # 亮度增加1.5倍，限制在0到255之间
    return bright_img  # 返回亮图像

# 创建低对比度图像
def create_low_contrast_image(img):  # 定义创建低对比度图像的函数
    low_contrast_img = np.clip((img - 128) * 0.5 + 128, 0, 255).astype(np.uint8)  # 减去128后乘以0.5再加128
    return low_contrast_img  # 返回低对比度图像

# 创建高对比度图像
def create_high_contrast_image(img):  # 定义创建高对比度图像的函数
    high_contrast_img = np.clip((img - 128) * 2 + 128, 0, 255).astype(np.uint8)  # 减去128后乘以2再加128
    return high_contrast_img  # 返回高对比度图像

# 主函数
def process_image_and_plot_histograms(image_path):  # 定义处理图像并绘制直方图的函数
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 读取图像并转换为灰度图像

    # 创建四类图像
    dark_img = create_dark_image(img)  # 创建暗图像
    bright_img = create_bright_image(img)  # 创建亮图像
    low_contrast_img = create_low_contrast_image(img)  # 创建低对比度图像
    high_contrast_img = create_high_contrast_image(img)  # 创建高对比度图像

    # 计算直方图图像
    original_hist_img = calc_histogram_image(img)  # 计算原始图像的直方图
    dark_hist_img = calc_histogram_image(dark_img)  # 计算暗图像的直方图
    bright_hist_img = calc_histogram_image(bright_img)  # 计算亮图像的直方图
    low_contrast_hist_img = calc_histogram_image(low_contrast_img)  # 计算低对比度图像的直方图
    high_contrast_hist_img = calc_histogram_image(high_contrast_img)  # 计算高对比度图像的直方图

    # 显示结果图像
    plt.figure(figsize=(10, 8))  # 创建绘图窗口

    # 显示原始图像的直方图图像
    plt.subplot(5, 2, 1)  # 创建第1个子图
    plt.imshow(img, cmap='gray')  # 显示原始图像
    plt.title('Original Image')  # 设置标题
    plt.axis('off')  # 关闭坐标轴

    plt.subplot(5, 2, 2)  # 创建第2个子图
    plt.imshow(original_hist_img)  # 显示原始直方图图像
    plt.title('Original Histogram')  # 设置标题
    plt.axis('off')  # 关闭坐标轴

    # 显示暗图像的直方图图像
    plt.subplot(5, 2, 3)  # 创建第3个子图
    plt.imshow(dark_img, cmap='gray')  # 显示暗图像
    plt.title('Dark Image')  # 设置标题
    plt.axis('off')  # 关闭坐标轴

    plt.subplot(5, 2, 4)  # 创建第4个子图
    plt.imshow(dark_hist_img)  # 显示暗直方图图像
    plt.title('Dark Histogram')  # 设置标题
    plt.axis('off')  # 关闭坐标轴

    # 显示亮图像的直方图图像
    plt.subplot(5, 2, 5)  # 创建第5个子图
    plt.imshow(bright_img, cmap='gray')  # 显示亮图像
    plt.title('Bright Image')  # 设置标题
    plt.axis('off')  # 关闭坐标轴

    plt.subplot(5, 2, 6)  # 创建第6个子图
    plt.imshow(bright_hist_img)  # 显示亮直方图图像
    plt.title('Bright Histogram')  # 设置标题
    plt.axis('off')  # 关闭坐标轴

    # 显示低对比度图像的直方图图像
    plt.subplot(5, 2, 7)  # 创建第7个子图
    plt.imshow(low_contrast_img, cmap='gray')  # 显示低对比度图像
    plt.title('Low Contrast Image')  # 设置标题
    plt.axis('off')  # 关闭坐标轴

    plt.subplot(5, 2, 8)  # 创建第8个子图
    plt.imshow(low_contrast_hist_img)  # 显示低对比度直方图图像
    plt.title('Low Contrast Histogram')  # 设置标题
    plt.axis('off')  # 关闭坐标轴

    # 显示高对比度图像的直方图图像
    plt.subplot(5, 2, 9)  # 创建第9个子图
    plt.imshow(high_contrast_img, cmap='gray')  # 显示高对比度图像
    plt.title('High Contrast Image')  # 设置标题
    plt.axis('off')  # 关闭坐标轴

    plt.subplot(5, 2, 10)  # 创建第10个子图
    plt.imshow(high_contrast_hist_img)  # 显示高对比度直方图图像
    plt.title('High Contrast Histogram')  # 设置标题
    plt.axis('off')  # 关闭坐标轴

    plt.tight_layout()  # 自动调整子图布局
    plt.show()  # 展示图像

#均衡化后的直方图输出

# 对图像进行直方图均衡化
def equalize_hist_image(img):  # 定义直方图均衡化函数
    equalized_img = cv2.equalizeHist(img)  # 使用OpenCV进行直方图均衡化
    return equalized_img  # 返回均衡化后的图像

# 主函数，包含直方图均衡化
def process_image_and_plot_histogramss(image_path):  # 定义处理图像并绘制均衡化直方图的函数
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # 读取图像并转换为灰度图像

    # 创建四类图像
    dark_img = create_dark_image(img)  # 创建暗图像
    bright_img = create_bright_image(img)  # 创建亮图像
    low_contrast_img = create_low_contrast_image(img)  # 创建低对比度图像
    high_contrast_img = create_high_contrast_image(img)  # 创建高对比度图像

    # 进行直方图均衡化
    equalized_img = equalize_hist_image(img)  # 进行直方图均衡化

    # 计算直方图图像
    original_hist_img = calc_histogram_image(img)  # 计算原始图像的直方图
    dark_hist_img = calc_histogram_image(dark_img)  # 计算暗图像的直方图
    bright_hist_img = calc_histogram_image(bright_img)  # 计算亮图像的直方图
    low_contrast_hist_img = calc_histogram_image(low_contrast_img)  # 计算低对比度图像的直方图
    high_contrast_hist_img = calc_histogram_image(high_contrast_img)  # 计算高对比度图像的直方图
    equalized_hist_img = calc_histogram_image(equalized_img)  # 计算均衡化图像的直方图

    # 显示结果图像
    plt.figure(figsize=(10, 10))  # 创建绘图窗口

    # 显示原始图像的直方图图像
    plt.subplot(6, 2, 1)  # 创建第1个子图
    plt.imshow(img, cmap='gray')  # 显示原始图像
    plt.title('Original Image')  # 设置标题
    plt.axis('off')  # 关闭坐标轴

    plt.subplot(6, 2, 2)  # 创建第2个子图
    plt.imshow(original_hist_img)  # 显示原始直方图图像
    plt.title('Original Histogram')  # 设置标题
    plt.axis('off')  # 关闭坐标轴

    # 显示暗图像的直方图图像
    plt.subplot(6, 2, 3)  # 创建第3个子图
    plt.imshow(dark_img, cmap='gray')  # 显示暗图像
    plt.title('Dark Image')  # 设置标题
    plt.axis('off')  # 关闭坐标轴

    plt.subplot(6, 2, 4)  # 创建第4个子图
    plt.imshow(dark_hist_img)  # 显示暗直方图图像
    plt.title('Dark Histogram')  # 设置标题
    plt.axis('off')  # 关闭坐标轴

    # 显示亮图像的直方图图像
    plt.subplot(6, 2, 5)  # 创建第5个子图
    plt.imshow(bright_img, cmap='gray')  # 显示亮图像
    plt.title('Bright Image')  # 设置标题
    plt.axis('off')  # 关闭坐标轴

    plt.subplot(6, 2, 6)  # 创建第6个子图
    plt.imshow(bright_hist_img)  # 显示亮直方图图像
    plt.title('Bright Histogram')  # 设置标题
    plt.axis('off')  # 关闭坐标轴

    # 显示低对比度图像的直方图图像
    plt.subplot(6, 2, 7)  # 创建第7个子图
    plt.imshow(low_contrast_img, cmap='gray')  # 显示低对比度图像
    plt.title('Low Contrast Image')  # 设置标题
    plt.axis('off')  # 关闭坐标轴

    plt.subplot(6, 2, 8)  # 创建第8个子图
    plt.imshow(low_contrast_hist_img)  # 显示低对比度直方图图像
    plt.title('Low Contrast Histogram')  # 设置标题
    plt.axis('off')  # 关闭坐标轴

    # 显示高对比度图像的直方图图像
    plt.subplot(6, 2, 9)  # 创建第9个子图
    plt.imshow(high_contrast_img, cmap='gray')  # 显示高对比度图像
    plt.title('High Contrast Image')  # 设置标题
    plt.axis('off')  # 关闭坐标轴

    plt.subplot(6, 2, 10)  # 创建第10个子图
    plt.imshow(high_contrast_hist_img)  # 显示高对比度直方图图像
    plt.title('High Contrast Histogram')  # 设置标题
    plt.axis('off')  # 关闭坐标轴

    # 显示直方图均衡化后的图像和直方图
    plt.subplot(6, 2, 11)  # 创建第11个子图
    plt.imshow(equalized_img, cmap='gray')  # 显示均衡化图像
    plt.title('Equalized Image')  # 设置标题
    plt.axis('off')  # 关闭坐标轴

    plt.subplot(6, 2, 12)  # 创建第12个子图
    plt.imshow(equalized_hist_img)  # 显示均衡化直方图图像
    plt.title('Equalized Histogram')  # 设置标题
    plt.axis('off')  # 关闭坐标轴

    plt.tight_layout()  # 自动调整子图布局
    plt.show()  # 展示图像



print("输入'1':图像灰度切片,把orange.jpg转为灰度图gray_image.jpg，然后切片")  # 打印功能1说明
print("输入'2':图像gray_image.jpg灰度图位平面切片")  # 打印功能2说明
print("输入'3':图像gray_image.jpg灰度图直方图统计")  # 打印功能3说明
print("输入'4':图像gray_image.jpg灰度图直方图统计,有均衡化过程")  # 打印功能4说明
num = input("请选择功能：")  # 获取用户输入功能选择

#这是灰度切片。选择了转化函数是一个范围全为白255
if num == '1':  # 如果选择功能1
    image = cv2.imread('orange.jpg', cv2.IMREAD_GRAYSCALE)  # 读取原始图像并转换为灰度图
    cv2.imwrite('gray_image.png', image)  # 保存灰度图像
    img = cv2.imread('orange.jpg',0)  # 再次读取原始图像
    fig = hui_du_ji_qie_pian(img)  # 执行灰度切片函数
    cv2.imwrite('hui_du_qie_pian.jpg',fig)  # 保存切片结果图像
    hua_T_r()  # 绘制转换函数图像

#灰度图的8个bit的位平面
if num == '2':  # 如果选择功能2
    wei_ping_mian_qie_pian()  # 执行位平面切片函数

#灰度图的直方图统计，暗图像，亮图像，低对比和高
if num == '3':  # 如果选择功能3
    image = 'orange.jpg'  # 设置图像路径
    process_image_and_plot_histograms(image)  # 执行处理图像并绘制直方图函数

#灰度图的均衡化后的直方图统计，暗图像，亮图像，低对比和高
if num == '4':  # 如果选择功能4
    image = 'orange.jpg'  # 设置图像路径
    process_image_and_plot_histogramss(image)  # 执行处理图像并绘制均衡化直方图函数
else:
    print("无效输入，请输入1到5之间的数字。")
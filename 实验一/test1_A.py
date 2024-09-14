import numpy as np
import matplotlib.pyplot as plt
import cv2

# 设置中文字体及负号显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用 SimHei 字体，解决中文显示问题
plt.rcParams['axes.unicode_minus'] = False  # 防止坐标轴负号显示不正常

# 灰度切片函数，将灰度值在 [190, 230] 范围内的像素变为白色（255）
def hui_du_qie_pian(img):
    h, w = img.shape  # 获取图像的高和宽
    # 创建一个新图像，初始化为全零
    new_img = np.zeros((h, w), dtype=np.uint8)

    # 将灰度值在 [190, 230] 范围内的像素设为 255（白色）
    new_img[(img >= 190) & (img <= 230)] = 255
    # 其他像素值保持不变
    new_img[(img < 190) | (img > 230)] = img[(img < 190) | (img > 230)]

    return new_img

# 绘制灰度切片转换函数 T(r)，其中 r 为灰度值，T 为转换后的灰度值
def hua_T_r():
    r = np.arange(0, 256)  # 生成灰度值范围 0 到 255
    T = np.where((r >= 190) & (r <= 230), 255, r)  # 如果灰度值在 [190, 230] 之间，T 为 255，否则 T 保持原值

    # 绘制转换函数 T(r)
    plt.plot(r, T, color='blue')
    plt.title('T(r) - 灰度级切片转换函数')  # 设置图像标题
    plt.xlabel('输入灰度值 r')  # 设置横坐标标签
    plt.ylabel('输出灰度值 T(r)')  # 设置纵坐标标签
    plt.grid(True)  # 添加网格线
    plt.savefig('T_r_function.png')  # 保存图像
    plt.show()

# 位平面切片函数，分解图像的每个位平面
def wei_ping_mian_qie_pian(img):
    # 创建一个列表，用于存储每个位平面的图像
    bit_planes = [(img >> i) & 1 for i in range(8)]  # 对每个像素进行位移操作，提取各个位平面

    # 设置展示窗口大小，2行4列的子图
    plt.figure(figsize=(10, 6))
    # 遍历每个位平面，显示对应的图像
    for i in range(8):
        plt.subplot(2, 4, i + 1)  # 创建子图
        plt.imshow(bit_planes[i] * 255, cmap='gray')  # 显示位平面图像，二值图像放大到 0-255 范围
        plt.title(f'Bit Plane {i}')  # 设置子图标题
        plt.axis('off')  # 隐藏坐标轴
    plt.tight_layout()  # 调整子图布局
    plt.savefig('bit_planes.png')  # 保存位平面图像
    plt.show()

# 计算图像的直方图，并返回绘制的直方图图像
def calc_histogram_image(image):
    # 计算图像的灰度直方图
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])

    # 创建绘图区域，显示直方图
    fig, ax = plt.subplots()
    ax.plot(hist, color='black')  # 绘制直方图
    ax.set_xlim([0, 256])  # 设置x轴范围
    ax.set_title('Histogram')  # 设置图像标题

    # 将绘制的直方图保存为图像
    fig.canvas.draw()

    # 使用 buffer_rgba() 代替 tostring_rgb()
    histogram_img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    # 因为 buffer_rgba 返回的是 RGBA 格式，所以要将其形状修改为 (宽, 高, 4)
    histogram_img = histogram_img.reshape(fig.canvas.get_width_height()[::-1] + (4,))

    # 去除透明度通道，保留 RGB 通道
    histogram_img = histogram_img[:, :, :3]  # 只保留前三个通道

    plt.close(fig)  # 关闭绘图，释放内存
    return histogram_img

# 创建暗图像，将图像整体亮度降低为原来的 50%
def create_dark_image(img):
    return np.clip(img * 0.5, 0, 255).astype(np.uint8)  # 防止越界后再将类型转换为 uint8

# 创建亮图像，将图像整体亮度提升为原来的 150%
def create_bright_image(img):
    return np.clip(img * 1.5, 0, 255).astype(np.uint8)  # 防止越界后再将类型转换为 uint8

# 创建低对比度图像
def create_low_contrast_image(img):
    return np.clip((img - 128) * 0.5 + 128, 0, 255).astype(np.uint8)  # 将像素差距缩小，降低对比度

# 创建高对比度图像
def create_high_contrast_image(img):
    return np.clip((img - 128) * 2 + 128, 0, 255).astype(np.uint8)  # 拉大像素差距，提高对比度

# 主要处理函数，处理输入图像，生成暗、亮、低对比度和高对比度的版本，并绘制直方图
def process_image_and_plot_histograms(img):
    # 生成各种图像的不同版本
    dark_img = create_dark_image(img)
    bright_img = create_bright_image(img)
    low_contrast_img = create_low_contrast_image(img)
    high_contrast_img = create_high_contrast_image(img)

    # 将图像及其标题存入列表
    images = [img, dark_img, bright_img, low_contrast_img, high_contrast_img]
    titles = ['Original Image', 'Dark Image', 'Bright Image', 'Low Contrast Image', 'High Contrast Image']

    # 创建子图显示每种图像及其直方图
    plt.figure(figsize=(10, 8))
    for i in range(5):
        # 显示每种图像
        plt.subplot(5, 2, 2 * i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])  # 设置标题
        plt.axis('off')  # 隐藏坐标轴

        # 显示每种图像对应的直方图
        plt.subplot(5, 2, 2 * i + 2)
        hist_img = calc_histogram_image(images[i])
        plt.imshow(hist_img)
        plt.title(f'{titles[i]} Histogram')  # 设置标题
        plt.axis('off')  # 隐藏坐标轴

    plt.tight_layout()  # 调整子图布局
    plt.savefig('image_histograms.png')  # 保存直方图结果
    plt.show()

# 功能选择菜单
print("输入'1': 图像灰度切片")
print("输入'2': 图像位平面切片")
print("输入'3': 图像直方图统计")
num = input("请选择功能：")  # 用户选择功能

# 根据选择执行不同的处理函数
if num == '1':
    img = cv2.imread('orange.jpg', cv2.IMREAD_GRAYSCALE)  # 读取灰度图
    sliced_img = hui_du_qie_pian(img)  # 灰度切片
    cv2.imwrite('sliced_image.png', sliced_img)  # 保存灰度切片后的图像
    hua_T_r()  # 绘制灰度切片转换函数
    # 显示切片后的图像
    plt.imshow(sliced_img, cmap='gray')
    plt.title('灰度切片图像')
    plt.axis('off')
    plt.show()

elif num == '2':
    img = cv2.imread('orange.jpg', cv2.IMREAD_GRAYSCALE)  # 读取灰度图
    wei_ping_mian_qie_pian(img)  # 位平面切片

elif num == '3':
    img = cv2.imread('orange.jpg', cv2.IMREAD_GRAYSCALE)  # 读取灰度图
    process_image_and_plot_histograms(img)  # 处理图像并显示不同对比度图像及直方图

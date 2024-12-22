import cv2
import os

# 创建结果保存目录
result_dir = "viola_jones_results"
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# 读取图像
image_path = "zmm.jpg"
image = cv2.imread(image_path)

# 转为灰度图
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 加载 OpenCV 提供的预训练 Haar 特征分类器，用于人脸检测
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# 使用 Viola-Jones 进行人脸检测
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# 标记检测到的人脸
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

# 保存结果
result_image_path = os.path.join(result_dir, "detected_faces.jpg")
cv2.imwrite(result_image_path, image)

# 显示检测结果
cv2.imshow("Detected Faces", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"人脸检测完成，结果保存至：{result_image_path}")

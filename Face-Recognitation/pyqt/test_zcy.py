import numpy as np
import cv2 as cv

class Face_Detect():
    def __init__(self):
        pass

    @staticmethod
    def face_detection(file_path):
        # 读取待检测的图片
        img = cv.imread(file_path)
        # print(img.shape)  # 打印图像的形状 (height, width, channels)

        # 缩小图像
        scale_percent = 50  # 缩小到原来的 50%
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        img_resized = cv.resize(img, (width, height))

        # 加载 Haar 级联分类器 预训练模型
        model_path = r"models\haarcascade_frontalface_alt2.xml"
        face_detector = cv.CascadeClassifier(model_path)  # <class 'cv2.CascadeClassifier'>

        # 检查是否成功加载模型
        if face_detector.empty():
            print("错误：模型文件未正确加载，请检查路径或文件是否存在")
        else:
            # 使用级联分类器检测人脸
            faces = face_detector.detectMultiScale(img_resized, scaleFactor=1.1, minNeighbors=1,
                                                minSize=(30, 30), maxSize=(300, 300))
            # cv.imwrite(fr'img_detect\figure_original.jpg', )
            # print(f"检测到 {len(faces)} 张人脸")  # 打印检测到的人脸数量
            # print(faces)  # 每张人脸的坐标和大小 (x, y, width, height)

            # 遍历每个检测到的人脸框
            for i, (x, y, width, height) in enumerate(faces):
                # 绘制人脸检测框
                cv.rectangle(img_resized, (x, y), (x + width, y + height), (0, 0, 255), 2, cv.LINE_8, 0)

                # 提取人脸框内的图片
                face_img = img_resized[y:y + height, x:x + width]

                # 显示提取的人脸图片
                # cv.imshow(f"Face {i + 1}", face_img)

                # 保存提取的人脸图片
                cv.imwrite(fr"detect\face_{i + 1}.jpg", face_img)  # 保存为 face_1.jpg, face_2.jpg, ...
            # 显示原图（带检测框）
            cv.imwrite(fr'img_detect\img_original.jpg', img_resized)
            print(i)
            # cv.imshow(fr'img_detect\img_original.jpg',faces)
            # cv.waitKey(0)  # 等待用户按键，0 表示无限等待
            # cv.destroyAllWindows()  # 关闭所有 OpenCV 窗口    
            return len(faces)
            
            
# if __name__ == '__main__':
#     file_name = r'img\family1.jpg'
#     num = Face_Detect.face_detection(file_name)
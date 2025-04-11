import numpy as np
import cv2
import os
from mtcnn.core.detect import create_mtcnn_net, MtcnnDetector
from mtcnn.core.vision import save_face, vis_face

# 加载已训练好的MTCNN模型
pnet, rnet, onet = create_mtcnn_net(p_model_path="./original_model/pnet_epoch.pt",
                                    r_model_path="./original_model/rnet_epoch.pt",
                                    o_model_path="./original_model/onet_epoch.pt", use_cuda=False)

# 建立人脸检测器
mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet, min_face_size=12)

# 实现数据集的人脸提取
def dataset_face_clip(mtcnn_detector):

    # 读取图像的路径
    path_read = r"D:\pythonProject\mtcnn-pytorch-master\datasets\ff_data"
    output_dir = r"D:\pythonProject\mtcnn-pytorch-master\datasets\ff"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.mkdir(os.path.join(output_dir, 'train'))
        os.mkdir(os.path.join(output_dir, 'val'))
        os.mkdir(os.path.join(output_dir, 'test'))

    for dataset_name in os.listdir(path_read):
        # aa是图片的全路径
        dataset_path = os.path.join(path_read, dataset_name)
        for classes_name in os.listdir(dataset_path):
            classes_path = os.path.join(dataset_path, classes_name)
            train_dir = os.path.join(output_dir, 'train', classes_name)
            val_dir = os.path.join(output_dir, 'val', classes_name)
            test_dir = os.path.join(output_dir, 'test', classes_name)

            if not os.path.exists(train_dir):
                os.mkdir(train_dir)
            if not os.path.exists(val_dir):
                os.mkdir(val_dir)
            if not os.path.exists(test_dir):
                os.mkdir(test_dir)

            for video_name in os.listdir(classes_path):
                video_dir = os.path.join(output_dir, dataset_name, classes_name, video_name)
                if not os.path.exists(video_dir):
                    os.mkdir(video_dir)
                framesum = 0
                for imgs in os.listdir(os.path.join(classes_path, video_name)):
                    img_path = os.path.join(classes_path, video_name, imgs)
                    img = cv2.imread(img_path)

                    img_shape = img.shape
                    img_height = img_shape[0]
                    img_width = img_shape[1]

                    dets, landmarks = mtcnn_detector.detect_face(img)

                    if dets.size == 0:
                        continue
                    else:
                        for i in range(dets.shape[0]):
                            bbox = dets[i, :4].astype('int')

                            # type(bbox)
                            Height = bbox[3] - bbox[1]
                            Width = bbox[2] - bbox[0]
                            if (Height > 0) and (Width > 0):
                                img_blank = np.zeros((Height, Width, 3), dtype=np.uint8)
                                # print(img_blank.shape)
                                #
                                # print(im_array.shape)
                                for h in range(Height):
                                    if bbox[1] + h >= img_height:
                                        continue
                                    for w in range(Width):
                                        if bbox[0] + w >= img_width:
                                            continue
                                        img_blank[h][w] = img[bbox[1] + h][bbox[0] + w]

                                # cv2.namedWindow("img_faces")  # , 2)
                                # cv2.imshow("img_faces", img_blank)  # 显示图片
                                img_blank = cv2.resize(img_blank, (256, 256), interpolation=cv2.INTER_CUBIC)
                                cv2.imwrite(filename=os.path.join(output_dir, dataset_name, classes_name, video_name,
                                                                  '0000{}.jpg'.format(str(framesum))), img=img_blank)
                                # cv2.imwrite(filename=os.path.join(r'D:\pythonProject\mtcnn-pytorch-master\results',
                                #                                   'img_face0{}.jpg'.format(str(i + 4))), img=img_blank)
                                # cv2.imwrite(r'D:\pythonProject\mtcnn-pytorch-master\results'+ "img_face_4" + str(i + 1) + ".jpg", img_blank)  # 将图片保存至你指定的文件夹
                                # cv2.waitKey(0)
                                framesum += 1
                            else:
                                continue

if __name__ == '__main__':
    # 加载已训练好的MTCNN模型
    pnet, rnet, onet = create_mtcnn_net(p_model_path="./original_model/pnet_epoch.pt",
                                        r_model_path="./original_model/rnet_epoch.pt",
                                        o_model_path="./original_model/onet_epoch.pt", use_cuda=False)

    # 建立人脸检测器
    mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet, min_face_size=12)
    dataset_face_clip(mtcnn_detector)


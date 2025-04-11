import cv2
from mtcnn.core.detect import create_mtcnn_net, MtcnnDetector
from mtcnn.core.vision import save_face, vis_face



if __name__ == '__main__':

    pnet, rnet, onet = create_mtcnn_net(p_model_path="./original_model/pnet_epoch.pt", r_model_path="./original_model/rnet_epoch.pt", o_model_path="./original_model/onet_epoch.pt", use_cuda=False)
    mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet, min_face_size=24)

    img = cv2.imread("./00002.jpg")
    # print(img.shape)
    # img_bg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # print(img_bg.shape)
    #b, g, r = cv2.split(img)
    #img2 = cv2.merge([r, g, b])

    bboxs, landmarks = mtcnn_detector.detect_face(img)
    print("****************")
    print(bboxs)
    print(bboxs.size)
    # print(bboxs)
    # print(landmarks)
    # print box_align
    save_name = 'r_4.jpg'
    save_face(img,bboxs,landmarks)

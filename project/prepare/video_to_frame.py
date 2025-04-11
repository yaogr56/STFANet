import os

from sklearn.model_selection import train_test_split
import cv2

# 将视频数据集划分train、val、te, 并建立相应的文件路径
def preprocess(self):
    if not os.path.exists(self.output_dir):
        os.mkdir(self.output_dir)
        os.mkdir(os.path.join(self.output_dir, 'train'))
        os.mkdir(os.path.join(self.output_dir, 'val'))
        os.mkdir(os.path.join(self.output_dir, 'te'))

    # Split train/val/testfile sets
    for classes in os.listdir(self.root_dir):
        classes_path = os.path.join(self.root_dir, classes)
        for file in os.listdir(classes_path):
            file_path = os.path.join(classes_path, file)
            video_files = [name for name in os.listdir(file_path)]

            train_and_valid, test = train_test_split(video_files, test_size=0.1, random_state=42)
            train, val = train_test_split(train_and_valid, test_size=0.2, random_state=42)

            train_dir = os.path.join(self.output_dir, 'train', file)
            val_dir = os.path.join(self.output_dir, 'val', file)
            test_dir = os.path.join(self.output_dir, 'te', file)

            if not os.path.exists(train_dir):
                os.mkdir(train_dir)
            if not os.path.exists(val_dir):
                os.mkdir(val_dir)
            if not os.path.exists(test_dir):
                os.mkdir(test_dir)

            for video in train:
                self.process_video(video, file, train_dir)

            for video in val:
                self.process_video(video, file, val_dir)

            for video in test:
                self.process_video(video, file, test_dir)

    print('Preprocessing finished.')

# 按照预处理划分好的数据集从视频提取帧，并保存为图片
def process_video(self, video, file_name, save_dir):
    # Initialize a VideoCapture object to read video data into a numpy array
    video_filename = video.split('.')[0]
    if not os.path.exists(os.path.join(save_dir, video_filename)):
        os.mkdir(os.path.join(save_dir, video_filename))
    for classes in os.listdir(self.root_dir):
        calsses_path = os.path.join(self.root_dir, classes)
        capture = cv2.VideoCapture(os.path.join(calsses_path, file_name, video))

        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Make sure splited video has at least 32 frames
        # EXTRACT_FREQUENCY = 5
        # if frame_count // EXTRACT_FREQUENCY <= 32:
        #     EXTRACT_FREQUENCY -= 1
        #     if frame_count // EXTRACT_FREQUENCY <= 32:
        #         EXTRACT_FREQUENCY -= 1
        #         if frame_count // EXTRACT_FREQUENCY <= 32:
        #             EXTRACT_FREQUENCY -= 1
        #             if frame_count // EXTRACT_FREQUENCY <= 32:
        #                 EXTRACT_FREQUENCY -= 1

        count = 0
        i = 0
        retaining = True

        while (count < frame_count and retaining):
            retaining, frame = capture.read()  # retaining 有没有读到图片   frame 当前截取一帧的图片
            if frame is None:
                continue

            if count < 50:
                if (frame_height != self.resize_height) or (frame_width != self.resize_width):
                    frame = cv2.resize(frame, (self.resize_width, self.resize_height))
                cv2.imwrite(filename=os.path.join(save_dir, video_filename, '0000{}.jpg'.format(str(i))), img=frame)
                i += 1
                count += 1
            else:
                capture.release()
                break

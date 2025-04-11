from config.path import Path

import os
from sklearn.model_selection import train_test_split

import torch
import cv2
import numpy as np
from torch.utils.data import Dataset


# class VideoProcess(object):
#     '''
#     视频切分为多个连续帧
#     '''
#     def __init__(self, dataset='ff', split='train', clip_len=32, preprocess=False):
#         self.root_dir, self.output_dir = Path.db_dir(dataset)
#         self.split = split
#         self.clip_len = clip_len
#
#
#         pass



class MyDataset(Dataset):
    '''
    自定义数据集
    '''

    # 注意第一次要预处理数据的
    def __init__(self, dataset='ff', split='train', clip_len=32, preprocess=False):
        self.root_dir, self.output_dir = Path.db_dir(dataset)
        folder = os.path.join(self.output_dir, split)
        self.clip_len = clip_len
        self.split = split

        # The following three parameters are chosen as described in the paper section 4.1
        self.resize_height = 256
        self.resize_width = 256
        self.crop_size = 224

        # if not self.check_integrity():
        #     raise RuntimeError('Dataset not found or corrupted.' +
        #                        ' You need to download it from official website.')

        # if (not self.check_preprocess()) or preprocess:
        #     print('Preprocessing of {} dataset, this will take long, but it will be done only once.'.format(dataset))
        #     # self.preprocess()
        if preprocess:
            print('Preprocessing of {} dataset, this will take long, but it will be done only once.'.format(dataset))
        # Obtain all the filenames of files inside all the class folders
        # Going through each class folder one at a time
        # classes2index = {'Deepfakes': 0, 'Face2Face': 0, 'FaceSwap': 0, 'NeuralTextures': 0, 'Origin': 1}
        if split == 'test':
            # classes2index = {'Celeb-real': 1, 'Celeb-synthesis': 0, 'YouTube-real': 1}
            # classes2index = {'Deepfakes': 0, 'Face2Face': 0, 'FaceSwap': 0, 'NeuralTextures': 0, 'Origin': 1}
            # classes2index = {'Deepfakes': 0, 'Face2Face': 0, 'FaceSwap': 0, 'NeuralTextures': 0, 'Origin': 1}
            # classes2index = {'Deepfakes': 0, 'Origin': 1}
            # classes2index = {'Celeb-real': 1, 'Celeb-synthesis': 0, 'YouTube-real': 1}
            classes2index = {'Deepfakes': 0, 'Origin': 1}
        self.fnames, labels = [], []
        for label in sorted(os.listdir(folder)):
            for fname in os.listdir(os.path.join(folder, label)):
                self.fnames.append(os.path.join(folder, label, fname))
                labels.append(classes2index[label])

        assert len(labels) == len(self.fnames)
        print('Number of {} videos: {:d}'.format(split, len(self.fnames)))

        self.label_array = np.array(labels)

        # Prepare a mapping between the label names (strings) and indices (ints)
        # self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}
        # Convert the list of label names into an array of label indices
        # self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)

        # if dataset == "ucf101":
        #     if not os.path.exists('dataloaders/ucf_labels.txt'):
        #         with open('dataloaders/ucf_labels.txt', 'w') as f:
        #             for id, label in enumerate(sorted(self.label2index)):
        #                 f.writelines(str(id+1) + ' ' + label + '\n')
        #
        # elif dataset == 'hmdb51':
        #     if not os.path.exists('dataloaders/hmdb_labels.txt'):
        #         with open('dataloaders/hmdb_labels.txt', 'w') as f:
        #             for id, label in enumerate(sorted(self.label2index)):
        #                 f.writelines(str(id+1) + ' ' + label + '\n')

    def __len__(self):
        return len(self.fnames)

    # 需要重写__getitem__方法
    def __getitem__(self, index):
        # Loading and preprocessing.
        buffer = self.load_frames(self.fnames[index])  # 一共有个文件夹
        buffer = self.crop(buffer, self.clip_len, self.crop_size)
        labels = np.array(self.label_array[index])

        # if self.split == 'train':
        #     # Perform data augmentation
        #     buffer = self.randomflip(buffer)
        buffer = self.normalize(buffer)
        buffer = self.to_tensor(buffer)
        # buffer = np.nan_to_num(buffer)
        return torch.from_numpy(buffer), torch.from_numpy(labels)

    # def check_integrity(self):
    #     if not os.path.exists(self.root_dir):
    #         return False
    #     else:
    #         return True

    def check_preprocess(self):
        # TODO: Check image size in output_dir
        if not os.path.exists(self.output_dir):
            return False
        elif not os.path.exists(os.path.join(self.output_dir, 'train')):
            return False

        for ii, video_class in enumerate(os.listdir(os.path.join(self.output_dir, 'train'))):
            for video in os.listdir(os.path.join(self.output_dir, 'train', video_class)):
                video_name = os.path.join(os.path.join(self.output_dir, 'train', video_class, video),
                                          sorted(
                                              os.listdir(os.path.join(self.output_dir, 'train', video_class, video)))[
                                              0])
                image = cv2.imread(video_name)
                if np.shape(image)[0] != 256 or np.shape(image)[1] != 256:
                    return False
                else:
                    break

            if ii == 10:
                break

        return True


    def randomflip(self, buffer):
        """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                frame = cv2.flip(buffer[i], flipCode=1)
                buffer[i] = cv2.flip(frame, flipCode=1)

        return buffer

    def imageresize(self, buffer):
        """resize each image from 124 to 256"""
        for i, frame in enumerate(buffer):
            frame = cv2.resize(buffer[i], (256, 256), interpolation=cv2.INTER_LINEAR)
            buffer[i] = frame
        return buffer

    def randomrotate(self, buffer):
        """Rotate 180 the given image and ground truth randomly with a probability of 0.5."""
        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                frame = cv2.rotate(buffer[i], cv2.ROTATE_180)
                buffer[i] = frame
        return buffer

    def normalize(self, buffer):
        for i, frame in enumerate(buffer):
            frame /= [[[255.0, 255.0, 255.0]]]
            buffer[i] = frame
        return buffer

    def to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))

    def load_frames(self, file_dir):
        # frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
        fra = sorted(([os.path.join(file_dir, img) for img in os.listdir(file_dir)]))
        # print(fra)
        frames = sorted(fra, key=lambda x: int(x.split("\\")[7][:-4]))
        frame_count = len(frames)
        buffer = np.empty((frame_count, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        for i, frame_name in enumerate(frames):
            frame = np.array(cv2.imread(frame_name)).astype(np.float64)
            buffer[i] = frame

        return buffer

    def crop(self, buffer, clip_len, crop_size):
        # randomly select time index for temporal jittering

        time_index = np.random.randint(buffer.shape[0] - clip_len)

        # Randomly select start indices in order to crop the video
        height_index = np.random.randint(buffer.shape[1] - crop_size)
        width_index = np.random.randint(buffer.shape[2] - crop_size)

        # Crop and jitter the video using indexing. The spatial crop is performed on
        # the entire array, so each frame is cropped in the same location. The temporal
        # jitter takes place via the selection of consecutive frames
        buffer = buffer[time_index:time_index + clip_len,
                 height_index:height_index + crop_size,
                 width_index:width_index + crop_size, :]

        return buffer


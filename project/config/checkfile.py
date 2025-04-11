import os
import shutil
from random import random
import shutil
from PIL.Image import Image

path_read = r"D:\pythonProject1\datasets\celeb_df"
count = 0
for dataset_class in sorted(os.listdir(path_read)):
    dataset_class_path = os.path.join(path_read, dataset_class)
    label_class_list = sorted(os.listdir(dataset_class_path))
    for label_class in label_class_list:
        label_class_path = os.path.join(dataset_class_path, label_class)
        video_list = sorted(os.listdir(label_class_path))
        for video_name in video_list:
            imgslist = os.listdir(os.path.join(label_class_path, video_name))
            print(len(imgslist))
            # if len(imgslist) <= 32:
            #     shutil.rmtree(os.path.join(label_class_path, video_name))



        count += len(video_list)
    print('{} 长度为: {}'.format(dataset_class, count))
    count = 0

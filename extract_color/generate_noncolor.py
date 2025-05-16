import numpy as np
import cv2
import os
from typing import Optional, Tuple, Union
from torch import Tensor

import torch


if __name__ == '__main__':
    train_ann_dir = "/data/sunzc/VisDrone2019/VisDrone2019-DET-train/annotations"

    image_dir = "/home/sunzc/VisDrone2019/visdrone_noncolor_image"

    save_dir_bbox = "/home/sunzc/VisDrone2019/train_anchor_normalize_noncolor"


    ann_files = os.listdir(train_ann_dir)
    image_names = os.listdir(image_dir)

    class_name_list = ['ignored regions', 'pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor', 'others']

    if not os.path.exists(save_dir_bbox):
        os.makedirs(save_dir_bbox)

    for image_name in image_names:

        if not image_name.endswith('.jpg'):
            continue
        ann_file = image_name.replace('.jpg', '.txt')

        train_ann_path = os.path.join(train_ann_dir, ann_file)

        image_path = os.path.join(image_dir, image_name)   
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_size = image.shape[:2]

        train_bbox = []
        generate_bbox = []
        new_size = []
        save_ann_path = os.path.join(save_dir_bbox, ann_file)


        # normalize anchor to [0, 1]
        # 把归一化后的bbox保存到esave_dir中
        with open(save_ann_path, 'w') as f_out:
                
            with open(train_ann_path, 'r') as f:
                for line in f.readlines():
                    line = line.strip().split(',')
                    x1, y1, w, h = line[:4]
                    bbox = [int(x1), int(y1), int(x1) + int(w), int(y1) + int(h)]
                    center_point = [(int(x1) + 0.5 * int(w)), (int(y1) + 0.5 * int(h))]
                    normal_center_point = [round(center_point[0]/image_size[1], 3), round(center_point[1]/image_size[0], 3)]
                    class_num = int(line[5])

                    if int(w) * int(h) <= 128:
                        continue

                    if class_num == 0 or class_num == 11:
                        continue
                    class_name = class_name_list[class_num]


                    if int(line[2]) * int(line[3]) < 256:
                        continue
                    class_num = int(line[5])
                    if class_num == 0 or class_num == 11:
                        continue


                    save_str = f"{class_name}: [{normal_center_point[0]}, {normal_center_point[1]}]"

                    f_out.write(save_str + '\n')

            


        # 把归一化后的bbox保存到save_dir中


        # with open(save_ann_path, 'w') as f_out:
        #     for bbox in train_bbox:
        #         bbox_str = ','.join(map(str, bbox))
        #         f_out.write(bbox_str + '\n')

        # print(f"Normalized bbox saved to {save_ann_path}")
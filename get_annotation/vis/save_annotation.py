import numpy as np
import cv2
import os
from typing import Optional, Tuple, Union
from torch import Tensor

import torch



if __name__ == '__main__':
    train_ann_dir = "/home/sunzc/VisDrone2019/annotations_new"

    image_dir = "/data/sunzc/VisDrone2019/VisDrone2019-DET-train/images/"

    save_dir_bbox = "/home/sunzc/VisDrone2019/visdrone_anchor_normalize_ann_new"


    ann_files = os.listdir(train_ann_dir)

    class_name_list = ['ignored regions', 'pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor', 'others']

    if not os.path.exists(save_dir_bbox):
        os.makedirs(save_dir_bbox)

    for ann_file in ann_files:

        if not ann_file.endswith('.txt'):
            continue

        train_ann_path = os.path.join(train_ann_dir, ann_file)

        image_path = os.path.join(image_dir, ann_file.split('.')[0] + '.jpg')   
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_size = image.shape[:2]

        train_bbox = []
        generate_bbox = []
        new_size = []
        save_ann_path = os.path.join(save_dir_bbox, ann_file)


        with open(save_ann_path, 'w') as f_out:
                
            with open(train_ann_path, 'r') as f:
                for line in f.readlines():
                    line = line.strip().split(',')
                    center_point = [int(line[0]) + 0.5*int(line[2]), int(line[1]) + 0.5*int(line[3])]
                    normal_center_point = [round(center_point[0]/image_size[1], 3), round(center_point[1]/image_size[0], 3)]
                    class_num = int(line[5])
                    class_name = class_name_list[class_num]

                    save_str = f"{class_name}: [{normal_center_point[0]}, {normal_center_point[1]}]"

                    f_out.write(save_str + '\n')

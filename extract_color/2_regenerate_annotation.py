# This script filters annotations by removing small bounding boxes (area < 144) 
# for vehicle classes (cars, vans, trucks, buses) and generates new annotation files
import torch
import shutil

import torch.nn as nn
import torchvision.models as models
from torchvision import transforms, utils
from PIL import Image
import os


# Input directories
image_root = "data/VisDrone2019-DET-val/images"  # Original images directory
ann_root = "data/VisDrone2019-DET-val/annotations"  # Original annotations directory

# Output directory for filtered annotations
save_new_annotation = "output/val_annotations_new"

if not os.path.exists(save_new_annotation):
    os.makedirs(save_new_annotation)

image_names = os.listdir(image_root)


for image_name in image_names:

    ann_name = image_name.split('.')[0] + '.txt'
    ann_path = os.path.join(ann_root, ann_name)
    lines_to_write = []
    with open(ann_path, "r") as f:
        num_line = 0
        tiny_line = 0
        tiny_car = 0
        car = 0

        for line in f.readlines():

            line = line.strip().split(',')
            class_num = int(line[5])

            x1, y1, w, h = line[:4]

            if class_num == 4 or class_num == 5 or class_num == 6 or class_num == 9 :
                if int(w) * int(h) > 144:
                    lines_to_write.append(','.join(line))
                else:
                    tiny_car += 1
            else:
                lines_to_write.append(','.join(line))

    save_ann_path = os.path.join(save_new_annotation, ann_name)
    with open(save_ann_path, "w") as f:
        for line in lines_to_write:
            f.write(line + '\n')



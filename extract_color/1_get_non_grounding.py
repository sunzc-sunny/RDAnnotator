# This script separates images into two categories:
# 1. Images suitable for color attribute analysis
# 2. Images not suitable for color analysis (non-grounding images)
import torch
import shutil

import torch.nn as nn
import torchvision.models as models
from torchvision import transforms, utils
from PIL import Image
import os


# Path to the original images and annotations
image_root = "data/VisDrone2019-DET-val/images"
ann_root = "data/VisDrone2019-DET-val/annotations"

# Output directories for categorized images
save_color_image = "output/visdrone_color_image_val"
save_non_grounding_image = "output/visdrone_non_grounding_image_val"


if not os.path.exists(save_color_image):
    os.makedirs(save_color_image)
if not os.path.exists(save_non_grounding_image):
    os.makedirs(save_non_grounding_image)


image_names = os.listdir(image_root)




for image_name in image_names:
    image_path = os.path.join(image_root, image_name)
    ann_name = image_name.split('.')[0] + '.txt'
    ann_path = os.path.join(ann_root, ann_name)
    image = Image.open(image_path).convert('RGB')
    average_pixel = tuple(map(int, image.resize((1, 1), Image.LANCZOS).getpixel((0, 0))))

    with open(ann_path, "r") as f:
        num_line = 0
        tiny_line = 0
        tiny_car = 0
        car = 0
        num_lines = len(f.readlines())

        if num_lines < 2:
            print(f"{image_name} has no object")
            shutil.copy(image_path, os.path.join(save_non_grounding_image, image_name))
            continue
        else:
            shutil.copy(image_path, os.path.join(save_color_image, image_name))
        # if sum(average_pixel) < 150:
        #     print(f"{image_name} is a night image")
        #     shutil.copy(image_path, os.path.join(save_non_color_image, image_name))
        #     continue




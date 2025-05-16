# This script extracts color information from objects in VisDrone dataset
# and generates annotations with color attributes
import cv2
import numpy as np
from sklearn.cluster import KMeans
import os
from concurrent.futures import ThreadPoolExecutor

# Dataset paths
train_dir = "data/VisDrone2019-DET-train/images/"  # Training images directory
ann_dir = "data/VisDrone2019-DET-train/annotations/"  # Original annotations directory

# Output directory for color annotations
save_dir_bbox = "output/visdrone_anchor_normalize_ann_color2"

if not os.path.exists(save_dir_bbox):
    os.makedirs(save_dir_bbox)


# Note: In OpenCV, HSV color ranges are:
# H (Hue): 0-179, S (Saturation): 0-255, V (Value): 0-255

preset_colors = {

    'dark': (0, 0, 40),

    'bright': (0, 0, 230)
}

class_name_list = ['ignored regions', 'pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor', 'others']


def color_save(image_name):
    color_names = []
    image_dir = os.path.join(train_dir, image_name)

    image = cv2.imread(image_dir)

    annotations_dir = os.path.join(ann_dir, image_name.split('.')[0] + '.txt')
    with open(annotations_dir, 'r') as f:
        for line in f.readlines():
            line = line.strip().split(',')
            if int(line[2]) * int(line[3]) < 256:
                continue
            class_num = int(line[5])
            if class_num == 0:
                continue

            bbox = [int(line[0])+5, int(line[1])+5, int(line[0])+int(line[2])-5, int(line[1])+int(line[3])-5]
            object_region = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            hsv_region = cv2.cvtColor(object_region, cv2.COLOR_BGR2HSV)
            hsv_region = hsv_region.reshape((-1, 3))
            k_means = KMeans(n_clusters=1, n_init=6)
            k_means.fit(hsv_region)

            main_color_hsv = k_means.cluster_centers_[0]
            main_color_rgb = cv2.cvtColor(np.uint8([[main_color_hsv]]), cv2.COLOR_HSV2RGB)[0][0]
            main_color_name = map_color_to_preset(main_color_hsv)
            color_names.append(main_color_name)

    return color_names



# Preset color definitions in HSV space
# These can be adjusted as needed


def map_color_to_preset(hsv_color):
    min_distance = float('inf')
    for color_name, preset_hsv in preset_colors.items():
        distance = ((hsv_color[0] - preset_hsv[0]) ** 2 + (hsv_color[1] - preset_hsv[1]) ** 2 + (hsv_color[2] - preset_hsv[2]) ** 2) ** 0.5
        if distance < min_distance:
            min_distance = distance
            main_color_name = color_name
    return main_color_name


def save_ann(image_name):
    image_dir = os.path.join(train_dir, image_name)
    image = cv2.imread(image_dir)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_size = image.shape[:2]
    print(image_size)
    print(image_name)
    color_names = color_save(image_name) 
    train_bbox = []
    generate_bbox = []
    new_size = []

    ann_name = image_name.split('.')[0] + '.txt'
    annotations_dir = os.path.join(ann_dir, image_name.split('.')[0] + '.txt')

    save_ann_path = os.path.join(save_dir_bbox, ann_name)
    i = 0
    with open(save_ann_path, 'w') as f_out:
            
        with open(annotations_dir, 'r') as f:
            for line in f.readlines():
                line = line.strip().split(',')
                if int(line[2]) * int(line[3]) < 256:
                    continue
                class_num = int(line[5])
                if class_num == 0:
                    continue
                center_point = [int(line[0]) + 0.5*int(line[2]), int(line[1]) + 0.5*int(line[3])]
                normal_center_point = [round(center_point[0]/image_size[1], 3), round(center_point[1]/image_size[0], 3)]
                class_num = int(line[5])
                class_name = class_name_list[class_num]

                save_str = f"{class_name}, {color_names[i]}: [{normal_center_point[0]}, {normal_center_point[1]}]"
                i += 1
                f_out.write(save_str + '\n')


if __name__ == '__main__':

    image_names = os.listdir(train_dir)


    for image_name in image_names:
        save_ann(image_name)


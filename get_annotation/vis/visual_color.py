import numpy as np
import cv2
import os
from typing import Optional, Tuple, Union
import re

import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor


def visualize_anchors(image, anchors, color, save_path=None):

    for anchor in anchors:
        x1, y1, x2, y2 = anchor
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)

    return image

def visualize_anchors2(image, anchors, color, save_path=None):

    for anchor in anchors:
        x1, y1, w, h = anchor
        x1 = int(x1)
        y1 = int(y1)
        w = int(w)
        h = int(h)
        cv2.rectangle(image, (x1, y1), (x1 + w, y1 + h), color, 1)


    return image


train_ann_dir = "/data/sunzc/VisDrone2019/VisDrone2019-DET-train/annotations/"

image_dir ="/data/sunzc/VisDrone2019/VisDrone2019-DET-train/10_1images"

ann_dir_bbox = "/data/sunzc/VisDrone2019/VisDrone2019-DET-train/10_1images"

save_dir_bbox = "/home/sunzc/VisDrone2019/10_1image_ann_colo_vis"

if not os.path.exists(save_dir_bbox):
    os.makedirs(save_dir_bbox)

preset_colors = {
    'red': (0, 168, 168), 
    'green': (50, 255, 255), 
    'blue': (100, 255, 255), 
    'yellow': (30, 255, 255),
    'black': (0, 0, 40),
    'white': (0, 0, 230)
}


def draw_points(image_name, points, color):

    image_path = os.path.join(image_dir, image_name)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for point in points:
        x, y = point

        cv2.circle(image, (int(x*image.shape[1]), int(y*image.shape[0])), 4, (255, 0, 0), 4)  

    save_path = os.path.join(save_dir_bbox, image_name.split('.')[0] + f"_{color}.jpg")
    plt.imshow(image)
    plt.savefig(save_path)
    plt.close()


def vis_samecolor_point(image_name):
    image_path = os.path.join(image_dir, image_name)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_size = image.shape[:2]

    color_ann_path = os.path.join(ann_dir_bbox, image_name.split('.')[0] + '.txt')

    red_points = []
    green_points = []
    blue_points = []
    yellow_points = []
    black_points = []
    white_points = []



    pattern = r"\[(\d+\.\d+), (\d+\.\d+)\]"  



    with open(color_ann_path, 'r') as f:
        for line in f.readlines():
            line_color_name = line.strip().split(' ')[1]

            matches = re.search(pattern, line) 
            center_point = [float(matches.group(1)), float(matches.group(2))]

            class_name = line_color_name

            if class_name == 'red:':
                red_points.append(center_point)
            elif class_name == 'green:':
                green_points.append(center_point)
            elif class_name == 'blue:':
                blue_points.append(center_point)
            elif class_name == 'yellow:':
                yellow_points.append(center_point)
            elif class_name == 'black:':
                black_points.append(center_point)
            elif class_name == 'white:':
                white_points.append(center_point)


        if len(red_points) > 0:
            draw_points(image_name, red_points, "red")
        if len(green_points) > 0:
            draw_points(image_name, green_points, "green")
        if len(blue_points) > 0:
            draw_points(image_name, blue_points, "blue")
        if len(yellow_points) > 0:
            draw_points(image_name, yellow_points, "yellow")
        if len(black_points) > 0:
            draw_points(image_name, black_points, "black")
        if len(white_points) > 0:
            draw_points(image_name, white_points, "white")




if __name__ == '__main__':

    train_dir = "/data/sunzc/VisDrone2019/VisDrone2019-DET-train/10_1images"


    image_names = os.listdir(train_dir)

    max_workers = 6  
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        executor.map(vis_samecolor_point, image_names)

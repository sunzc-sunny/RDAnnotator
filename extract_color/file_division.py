# 2点来判断这个图是否要使用颜色属性
# 1.图像中小的物体是不是个数很多，有没有超过半数
# 2.图像是不是夜间拍摄的



import torch
import shutil

import torch.nn as nn
import torchvision.models as models
from torchvision import transforms, utils
from PIL import Image
import os


image_root = "/data/sunzc/VisDrone2019/VisDrone2019-DET-val/images"
ann_root = "/home/sunzc/VisDrone2019/val_annotations_new"


save_non_color_image = "/home/sunzc/VisDrone2019/visdrone_non_color_image_val"
save_color_image = "/home/sunzc/VisDrone2019/visdrone_color_image_val"

save_tiny_cars_image = "/home/sunzc/VisDrone2019/visdrone_non_color_image_val"

save_non_grounding_image = "/home/sunzc/VisDrone2019/visdrone_non_grounding_image_val"

if not os.path.exists(save_non_color_image):
    os.makedirs(save_non_color_image)
if not os.path.exists(save_color_image):
    os.makedirs(save_color_image)
if not os.path.exists(save_non_grounding_image):
    os.makedirs(save_non_grounding_image)
if not os.path.exists(save_tiny_cars_image):
    os.makedirs(save_tiny_cars_image)

image_names = os.listdir(image_root)

save_dir = "/data/sunzc/VCoR/my_train"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)



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


        if sum(average_pixel) < 150:
            print(f"{image_name} is a night image")
            shutil.copy(image_path, os.path.join(save_non_color_image, image_name))
            continue

        for line in f.readlines():

            line = line.strip().split(',')
            class_num = int(line[5])
            if class_num == 0 or class_num == 11:
                continue
            num_line += 1
            x1, y1, w, h = line[:4]

            if class_num == 4 or class_num == 5 or class_num == 6 or class_num == 9 :
                car += 1
                if int(w) * int(h) <= 256:
                    tiny_car += 1

            if int(w) * int(h) <= 256:
                tiny_line += 1

        # if tiny_line > 2 * num_line / 3:
        #     print(f"{image_name} has too many tiny cars")
        #     shutil.copy(image_path, os.path.join(save_non_color_image, image_name))
        # else:
        #     shutil.copy(image_path, os.path.join(save_color_image, image_name))
        # print(tiny_car, car )
        if num_line == 0 or num_line == 1:
            print(f"{image_name} has no object")
            shutil.copy(image_path, os.path.join(save_non_grounding_image, image_name))
        elif tiny_car >  car / 2:
            print(f"{image_name} has too many tiny cars")
            shutil.copy(image_path, os.path.join(save_tiny_cars_image, image_name))
        

                # print(f"{image_name}: Predicted: {color_mapping[predicted.item()]}, [{normal_center_point[0]}, {normal_center_point[1]}]")

    # exit()
            # x1, y1, w, h = int(x1), int(y1), int(w), int(h)
            # class_name = line[5]
            # color_name = line[6]
            # center_point = [(x1 + w)/2, (y1 + h)/2]
            # show_str = class_name + "_" + color_name + "_" + str(center_point[0]) + "_" + str(center_point[1])



# total_correct = 0
# total_samples = 0
# with torch.no_grad():
#     for images, labels, image_name in test_loader:
#         images = images.to(device)
#         labels = labels.to(device)
#         outputs = vgg16(images)
#         _, predicted = torch.max(outputs.data, 1)
#         _, labels = torch.max(labels, 1)
#         total_samples += labels.size(0)
#         total_correct += (predicted == labels).sum().item()
#         print(f"{image_name}: Predicted: {color_mapping[predicted.item()]}, Ground Truth: {color_mapping[labels.item()]}")

# # 打印准确率
# accuracy = total_correct / total_samples
# print(f"Test Accuracy: {accuracy*100:.2f}%")



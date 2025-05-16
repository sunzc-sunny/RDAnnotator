
import torch

import torch.nn as nn
import torchvision.models as models
from vocr_dataset import VCoR
from torchvision import transforms, utils
from PIL import Image
import os

class CustomWideResNet101(nn.Module):
    def __init__(self, num_classes=6):
        super(CustomWideResNet101, self).__init__()
        # Load pre-trained WideResNet50-2 model
        self.wide_resnet = models.wide_resnet101_2(pretrained=True)
        
        # Modify the final fully connected layer to output 6 classes
        in_features = self.wide_resnet.fc.in_features
        self.wide_resnet.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.wide_resnet(x)

model = CustomWideResNet101()


checkpoint = torch.load('wide_resnet_color_model_99_89.80.pth')
model.load_state_dict(checkpoint)


model.eval()



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


color_mapping = {
    0: 'black', 1: 'blue', 2: 'green', 3: 'red', 4: 'white', 5: 'yellow'
}

transform = transforms.Compose([
    transforms.Resize(size=(32, 32), interpolation=Image.BICUBIC),

    transforms.ToTensor(),

    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
])


image_root = "VisDrone2019/all_images"

ann_root = "VisDrone2019/all_annotations"

save_dir_bbox = "VisDrone2019/val_image_color"


if not os.path.exists(save_dir_bbox):
    os.makedirs(save_dir_bbox)

class_name_list = ['ignored regions', 'pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor', 'others']

image_names = os.listdir(image_root)

save_dir = "output/train_wideresnet101"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for color_name in color_mapping.values():
    color_dir = os.path.join(save_dir, color_name)
    if not os.path.exists(color_dir):
        os.makedirs(color_dir)


for image_name in image_names:
    if image_name.split('.')[-1] != 'jpg':
        continue
    image_path = os.path.join(image_root, image_name)
    ann_name = image_name.split('.')[0] + '.txt'
    ann_path = os.path.join(ann_root, ann_name)
    image = Image.open(image_path).convert('RGB')
    image_size = image.size

    save_ann_path = os.path.join(save_dir_bbox, ann_name)

    with open(save_ann_path, "w") as f_out:

        with open(ann_path, "r") as f:
            for line in f.readlines():
                line = line.strip().split(',')
                x1, y1, w, h = line[:4]
                bbox = [int(x1)-4, int(y1)-4, int(x1) + int(w)-4, int(y1) + int(h)-4]
                center_point = [(int(x1) + 0.5 * int(w)), (int(y1) + 0.5 * int(h))]
                normal_center_point = [round(center_point[0]/image_size[0], 3), round(center_point[1]/image_size[1], 3)]
                class_num = int(line[5])

                if int(w) * int(h) <= 128:
                    continue

                if class_num == 0 or class_num == 11:
                    continue

                class_name = class_name_list[class_num]
                object_region = image.crop(bbox)

                object_region = transform(object_region)

                
                object_region = object_region.unsqueeze(0).to(device)
                outputs = model(object_region)
                _, predicted = torch.max(outputs.data, 1)

                save_str = f"{class_name}, {color_mapping[predicted.item()]}: [{normal_center_point[0]}, {normal_center_point[1]}]"


                f_out.write(save_str + "\n")




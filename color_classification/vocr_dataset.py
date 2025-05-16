import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch.optim as optim
import os
from matplotlib import pyplot as plt
import random


class VCoR(torch.utils.data.Dataset):
    def __init__(self, root, img_size=(32, 32)):

        self.root = root
        self.img_size = img_size

 
        transform = transforms.Compose([
            transforms.Resize(size=(32, 32), interpolation=Image.BICUBIC),  

            transforms.ToTensor(),

            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])
        self.transforms = transform
        self.image_labels = {}  # Dictionary to hold image labels
        self.image_names = {}
        self.load_data_list()
        self.load_label_list()


    def load_data_list(self):
        self.black_names = os.listdir(os.path.join(self.root, 'black'))
        self.black_names = [os.path.join(self.root, 'black', name) for name in self.black_names if name.endswith('.jpg')]

        self.grey_names = os.listdir(os.path.join(self.root, 'grey'))
        self.grey_names = [os.path.join(self.root, 'grey', name) for name in self.grey_names if name.endswith('.jpg')]

        self.black_names = self.black_names + self.grey_names

        self.blue_names = os.listdir(os.path.join(self.root, 'blue'))
        self.blue_names = [os.path.join(self.root, 'blue', name) for name in self.blue_names if name.endswith('.jpg')]


        self.green_names = os.listdir(os.path.join(self.root, 'green'))
        self.green_names = [os.path.join(self.root, 'green', name) for name in self.green_names if name.endswith('.jpg')]

        self.red_names = os.listdir(os.path.join(self.root, 'red'))
        self.red_names = [os.path.join(self.root, 'red', name) for name in self.red_names if name.endswith('.jpg')]


        self.orange_names = os.listdir(os.path.join(self.root, 'orange'))
        self.orange_names = [os.path.join(self.root, 'orange', name) for name in self.orange_names if name.endswith('.jpg')]


        self.pink_names = os.listdir(os.path.join(self.root, 'pink'))
        self.pink_names = [os.path.join(self.root, 'pink', name) for name in self.pink_names if name.endswith('.jpg')]


        self.red_names = self.red_names + self.orange_names + self.pink_names

        self.white_names = os.listdir(os.path.join(self.root, 'white'))
        self.white_names = [os.path.join(self.root, 'white', name) for name in self.white_names if name.endswith('.jpg')]

        self.silver_names = os.listdir(os.path.join(self.root, 'silver'))
        self.silver_names = [os.path.join(self.root, 'silver', name) for name in self.silver_names if name.endswith('.jpg')]

        self.white_names = self.white_names + self.silver_names

        self.yellow_names = os.listdir(os.path.join(self.root, 'yellow'))
        self.yellow_names = [os.path.join(self.root, 'yellow', name) for name in self.yellow_names if name.endswith('.jpg')]

        self.gold_names = os.listdir(os.path.join(self.root, 'gold'))
        self.gold_names = [os.path.join(self.root, 'gold', name) for name in self.gold_names if name.endswith('.jpg')]

        self.yellow_names = self.yellow_names + self.gold_names

        
    
        self.image_names['black'] = self.black_names
        self.image_names['blue'] = self.blue_names
        self.image_names['green'] = self.green_names
        self.image_names['red'] = self.red_names
        self.image_names['white'] = self.white_names
        self.image_names['yellow'] = self.yellow_names


    def load_label_list(self):
        color_mapping = {
            'black': 0, 'blue': 1, 'green': 2, 'red': 3, 'white': 4, 'yellow': 5
        }

        for color, names in self.image_names.items():


            for name in names:
                self.image_labels[str(name)] = color_mapping[color]


    def __getitem__(self, index):
        image_name = list(self.image_labels.keys())[index]
        img = Image.open(image_name).convert('RGB')

        label = self.image_labels[image_name]
        # 转化为one-hot编码
        label = F.one_hot(torch.tensor(label), num_classes=6)
        img = self.transforms(img)
        return img, label, image_name

    def __len__(self):
        return len(self.image_labels)


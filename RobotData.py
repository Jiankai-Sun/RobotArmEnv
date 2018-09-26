import pandas as pd
import numpy as np
from PIL import Image
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets

class RobotDataset(Dataset):
    def __init__(self, img_path='/home/data/RobotArm-DA', transform):
        """
        Args:
            img_path (string): path to the folder where images are, /home/data/RobotArm-DA
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Transforms
        self.transform = transform
        self.img_path = img_path
        sub_folder = os.listdir(self.img_path)
        print(sub_folder)
        self.rgb_list = []
        self.depth_list = []
        for sub in sub_folder:
            rgb_sub_foler = os.path.join(self.img_path, sub, '/RGB')
            self.rgb_list.append(os.path.join(rgb_sub_foler, img_file) for img_file in os.listdir(rgb_sub_foler))
            depth_sub_foler = os.path.join(self.img_path, sub, '/depth')
            self.depth_list.append(os.path.join(depth_sub_foler, depth_file) for depth_file in os.listdir(depth_sub_foler))
        
        assert len(self.rgb_list) == len(self.depth_list), 'RGB != Depth'
        self.data_len = len(self.rgb_list)
        

    def __getitem__(self, index):
        rgb_name = self.rgb_list[index]
        depth_name = self.depth_list[index]
        # Open image
        rgb_img = Image.open(rgb_name)
        depth_arr = numpy.load(depth_name) / 1000. # millimeter -> meter

        # If there is an operation
        if self.transform:
            rgb_img = self.transform(rgb_img)

        return rgb_img, depth_arr

    def __len__(self):
        return self.data_len

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    custom_RobotDataset =  RobotDataset(img_path='/home/data/RobotArm-DA'，
                                        transform=transforms.Compose([
                                                transforms.Resize([128, 128]),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5, 0.5, 0.5),
                                                                    (0.5, 0.5, 0.5))
                                            ]))
    RobotDataset_loader = torch.utils.data.DataLoader(dataset=custom_RobotDataset,
                                                    batch_size=10,
                                                    shuffle=False)
    for i, rgb_img, depth in enumerate(RobotDataset_loader):
        plt.imshow(rgb_img)
        plt.show()
        plt.imshow(depth)
        plt.show()
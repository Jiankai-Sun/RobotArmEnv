import numpy as np
from PIL import Image
import os

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data.dataset import Dataset  # For custom datasets

class RobotDataset(Dataset):
    def __init__(self, img_path='/home/data/RobotArm-DA', label_path='/home/data1/full/Group_32_a', transform=None):
        """
        Args:
            img_path (string): path to the folder where images are, /home/data/RobotArm-DA
            transform: pytorch transforms for transforms and tensor conversion
        This dataset ignore the temporal information of each subdir, e.g. 0000.
        """
        # Transforms
        self.transform = transform
        self.img_path = img_path
        self.label_path = label_path
        sub_folder = sorted([os.path.join(img_path, o) for o in os.listdir(img_path) 
                    if os.path.isdir(os.path.join(img_path, o))])
        label_sub_folder = sorted([os.path.join(label_path, o) for o in os.listdir(label_path) 
                    if os.path.isdir(os.path.join(label_path, o))])
        self.rgb_list = []
        self.depth_list = []
        self.semantic_label_list = []
        for sub in label_sub_folder:
            sub_foler = os.path.join(sub, 'RGB')
            self.semantic_label_list.extend([os.path.join(sub_foler, semantic_label_folder) for semantic_label_folder in sorted(os.listdir(sub_foler)) if os.path.isdir(semantic_label_folder)])
            
        for sub in sub_folder:
            rgb_sub_foler = os.path.join(sub, 'RGB')
            self.rgb_list.extend([os.path.join(rgb_sub_foler, img_file) for img_file in sorted(os.listdir(rgb_sub_foler)) if img_file.split('.')[-1] == 'jpg'])
            depth_sub_foler = os.path.join(sub, 'depth')
            self.depth_list.extend([os.path.join(depth_sub_foler, depth_file) for depth_file in sorted(os.listdir(depth_sub_foler)) if depth_file.split('.')[-1] == 'npy'])
        print('Number of RGB images: ', len(self.rgb_list))
        print('Number of Depth images: ', len(self.depth_list))
        print('Number of Label images: ', len(self.semantic_label_list))

        assert len(self.rgb_list) == len(self.depth_list), 'RGB != Depth'
        self.data_len = len(self.rgb_list)

    def __getitem__(self, index):
        rgb_name = self.rgb_list[index]
        depth_name = self.depth_list[index]
        semantic_label_name = os.path.join(self.semantic_label[index], 'direct_mask.png')
        print('rgb_name: ', rgb_name, 'depth_name: ', depth_name)
        # Open image
        rgb_img = Image.open(rgb_name)
        semantic_label = Image.open(semantic_label_name)
        depth_arr = np.load(depth_name) / 1000. # millimeter -> meter

        # If there is an operation
        if self.transform:
            rgb_img = self.transform(rgb_img)

        return rgb_img, depth_arr, semantic_label

    def __len__(self):
        return self.data_len

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    custom_RobotDataset =  RobotDataset(img_path='/home/data/RobotArm-DA',
                                        label_path='/home/data1/full/Group_32_a',
                                        transform=transforms.Compose([
                                                transforms.Resize([128, 128]),
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.5, 0.5, 0.5),
                                                                    (0.5, 0.5, 0.5))
                                            ]))
    RobotDataset_loader = torch.utils.data.DataLoader(dataset=custom_RobotDataset,
                                                    batch_size=10,
                                                    shuffle=False)
    for i, (rgb_img, depth) in enumerate(RobotDataset_loader):
        plt.imshow(rgb_img[0].permute(1, 2, 0))
        plt.show()
        plt.imshow(depth[0])
        plt.show()
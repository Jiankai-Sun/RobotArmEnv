import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

class RobotNet(nn.Module):

    def __init__(self, feature_dim, object_num):
        super(RobotNet, self).__init__()
        self.fc1 = nn.Linear(feature_dim + 4 * object_num, 4)

    def forward(self, feature, bbox):
        x = torch.cat((feature, bbox), 1)
        x = self.fc1(x)
        return x
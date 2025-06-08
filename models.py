from PIL import Image
from os.path import join
import imageio
from torch import nn
from torch.nn.modules.linear import Linear
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.models as models
import math


class EncoderFemnist(nn.Module):
    def __init__(self, code_length):
        super(EncoderFemnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10,20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(int(320), code_length)
        
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        z = F.relu(self.fc1(x))
        return z       


class CNNFemnist(nn.Module):
    def __init__(self, args,code_length=50,num_classes = 62):
        super(CNNFemnist, self).__init__()
        self.code_length = code_length
        self.num_classes = num_classes
        self.feature_extractor = EncoderFemnist(self.code_length)
        self.classifier = nn.Sequential(nn.Dropout(0.2),
                                        nn.Linear(self.code_length, self.num_classes),
                                        nn.LogSoftmax(dim=1))
        
    def forward(self, x):
        z = self.feature_extractor(x)
        p = self.classifier(z)
        return z,p
       
"""        
class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.num_classes = num_classes
        self.model = models.resnet18(pretrained=False)
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(512, self.num_classes)
        #self.num_classes = num_classes
        #self.model = models.resnet18(num_classes=self.num_classes)

    def forward(self, x):
        out = self.model(x)
        return out
"""


class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.num_classes = num_classes
        self.model = models.resnet18(pretrained=False)

        # 修改第一个卷积层
        self.model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        # 替换 BatchNorm 层为 GroupNorm
        self.model.bn1 = nn.GroupNorm(num_groups=32, num_channels=64)
        self.model.layer1[0].bn1 = nn.GroupNorm(num_groups=32, num_channels=64)
        self.model.layer1[0].bn2 = nn.GroupNorm(num_groups=32, num_channels=64)
        self.model.layer1[1].bn1 = nn.GroupNorm(num_groups=32, num_channels=64)
        self.model.layer1[1].bn2 = nn.GroupNorm(num_groups=32, num_channels=64)

        self.model.layer2[0].bn1 = nn.GroupNorm(num_groups=32, num_channels=128)
        self.model.layer2[0].bn2 = nn.GroupNorm(num_groups=32, num_channels=128)
        self.model.layer2[1].bn1 = nn.GroupNorm(num_groups=32, num_channels=128)
        self.model.layer2[1].bn2 = nn.GroupNorm(num_groups=32, num_channels=128)

        self.model.layer3[0].bn1 = nn.GroupNorm(num_groups=32, num_channels=256)
        self.model.layer3[0].bn2 = nn.GroupNorm(num_groups=32, num_channels=256)
        self.model.layer3[1].bn1 = nn.GroupNorm(num_groups=32, num_channels=256)
        self.model.layer3[1].bn2 = nn.GroupNorm(num_groups=32, num_channels=256)

        self.model.layer4[0].bn1 = nn.GroupNorm(num_groups=32, num_channels=512)
        self.model.layer4[0].bn2 = nn.GroupNorm(num_groups=32, num_channels=512)
        self.model.layer4[1].bn1 = nn.GroupNorm(num_groups=32, num_channels=512)
        self.model.layer4[1].bn2 = nn.GroupNorm(num_groups=32, num_channels=512)

        # 替换 downsample 中的 BatchNorm 为 GroupNorm
        if hasattr(self.model.layer2[0], 'downsample'):
            self.model.layer2[0].downsample[1] = nn.GroupNorm(num_groups=32, num_channels=128)
        if hasattr(self.model.layer3[0], 'downsample'):
            self.model.layer3[0].downsample[1] = nn.GroupNorm(num_groups=32, num_channels=256)
        if hasattr(self.model.layer4[0], 'downsample'):
            self.model.layer4[0].downsample[1] = nn.GroupNorm(num_groups=32, num_channels=512)

        # 去掉 maxpool 层
        self.model.maxpool = nn.Identity()

        # 修改最后的全连接层
        self.model.fc = nn.Linear(512, self.num_classes)

    def forward(self, x):
        out = self.model(x)
        return out



class Smallnet(nn.Module):
    def __init__(self, num_classes=10):
        super(Smallnet, self).__init__()
        self.num_classes = num_classes
        self.model = models.mobilenet_v2(pretrained=False)
        self.model.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.model.last_channel, self.num_classes),
        )

    def forward(self, x):
        out = self.model(x)
        return out



class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()
        self.num_classes = num_classes
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(128, self.num_classes, bias=True),
        )

    def forward(self, x):
        out = self.model(x)
        return out


class MLP(nn.Module):
    def __init__(self, num_classes=10):
        super(MLP, self).__init__()
        self.num_classes = num_classes
        self.model = nn.Sequential(
            nn.Linear(600, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.num_classes)
        )

    def forward(self, x):
        out = self.model(x)
        return out


class ShuffLeNet(nn.Module):
    def __init__(self, args,code_length=64,num_classes = 10):
        super(ShuffLeNet, self).__init__()
        self.code_length = code_length
        self.num_classes = num_classes  
        self.feature_extractor = models.shufflenet_v2_x1_0(num_classes=self.code_length)
        self.classifier =  nn.Sequential(
                                nn.Linear(self.code_length, self.num_classes))
    def forward(self,x):
        z = self.feature_extractor(x)
        p = self.classifier(z)
        return z,p





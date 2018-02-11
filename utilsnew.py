# -*- coding: utf-8 -*-
"""Implements some utils

TODO:
"""
from torch.utils.data import DataLoader,Dataset
import numpy as np
import random
from PIL import Image
import torch
import os
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

import argparse
import random

from torchvision import transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

class Visualizer2:
    def __init__(self, show_step=10, image_size=(100,100)):
        self.transform = transforms.Compose([transforms.Normalize(mean = [-2.118, -2.036, -1.804], #Equivalent to un-normalizing ImageNet (for correct visualization)
                                            std = [4.367, 4.464, 4.444]),
                                            transforms.ToPILImage(),
                                            transforms.Resize(image_size)])

        self.show_step = show_step
        self.step = 0
        self.figure, (self.lr_plot, self.fake_plot) = plt.subplots(1,2)
        self.figure.show()
        self.lr_image_ph = None
        self.fake_hr_image_ph = None

    def show(self, inputsG, inputstrans,count,name):
        for j in range(2):
            i = random.randint(0, inputsG.size(0) -1)
            lr_image = self.transform(inputsG[i])
            transformimage = self.transform(inputstrans[i])
            if self.lr_image_ph is None:
                self.lr_image_ph = self.lr_plot.imshow(lr_image)
                self.fake_hr_image_ph = self.fake_plot.imshow(transformimage)
            else:
                self.lr_image_ph.set_data(lr_image)
                self.fake_hr_image_ph.set_data(transformimage)

            self.figure.canvas.draw()
            self.figure.savefig('./resultimages/' +name +'/'+ str(count)+'.png')
            count+=1
        return count

class SingleImage(Dataset):
    
    def __init__(self, imageFolder, transform=None):
        self.imageFolder = imageFolder 
        self.transform = transform
        
    def __getitem__(self,index):
        imgname = random.choice(os.listdir(self.imageFolder))
        while not(imgname[-1] =='g'):
            imgname = random.choice(os.listdir(self.imageFolder))
        img = Image.open(os.path.join(self.imageFolder,imgname))

        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(os.listdir(self.imageFolder))

# -*- coding: utf-8 -*-
"""Implements some utils

TODO:
"""

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
        #self.hr_image_ph = None
        self.fake_hr_image_ph = None

    def show(self, inputsG, inputstrans,count,name):

        # self.step += 1
        # if self.step == self.show_step:
        # self.step = 0

        for j in range(1):
            i = random.randint(0, inputsG.size(0) -1)
            lr_image = self.transform(inputsG[i])
            #hr_image = self.transform(inputsD_real[i])
            transformimage = self.transform(inputstrans[i])

            if self.lr_image_ph is None:
                self.lr_image_ph = self.lr_plot.imshow(lr_image)
                #self.hr_image_ph = self.hr_plot.imshow(hr_image)
                self.fake_hr_image_ph = self.fake_plot.imshow(transformimage)
            else:
                self.lr_image_ph.set_data(lr_image)
                #self.hr_image_ph.set_data(hr_image)
                self.fake_hr_image_ph.set_data(transformimage)

            self.figure.canvas.draw()
            self.figure.savefig('./resultimages/' +name +'/'+ str(count)+'.png')
            count+=1
        return count


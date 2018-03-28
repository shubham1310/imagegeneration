import random
from PIL import Image
import os
import matplotlib.pyplot as plt
from torchvision import transforms

class Visualizer2:
    def __init__(self, image_size=(100,100)):
        self.transform = transforms.Compose(
                            [transforms.Normalize(mean = [-2.118, -2.036, -1.804], #Equivalent to un-normalizing ImageNet (for correct visualization)
                            std = [4.367, 4.464, 4.444]),
                            transforms.ToPILImage(),
                            transforms.Resize(image_size)]
                            )
        self.step = 0
        self.figure, (self.lr_plot, self.fake_plot) = plt.subplots(1,2)
        self.figure.show()

    def show(self, inputsG, inputstrans,count,name):
        i = random.randint(0, inputsG.size(0) -1)
        self.lr_plot.imshow(self.transform(inputsG[i]))
        self.fake_plot.imshow(self.transform(inputstrans[i]))
        self.figure.canvas.draw()

        images = [self.transform(inputsG[i]), self.transform(inputstrans[i])]
        widths, heights = zip(*(i.size for i in images))
        total_width = sum(widths)
        max_height = max(heights)

        new_im = Image.new('RGB', (total_width, max_height))

        x_offset = 0
        for im in images:
          new_im.paste(im, (x_offset,0))
          x_offset += im.size[0]

        # new_im.save('./resultimages/' +name + str(count)+'.jpg')
        self.figure.savefig('./resultimages/' +name + str(count)+'.png')
        return count + 1

class SingleImage():
    
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

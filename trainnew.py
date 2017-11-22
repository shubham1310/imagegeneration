import matplotlib
# import matplotlib.pyplot as plt
matplotlib.use('Agg')

import argparse
import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from tensorboard_logger import configure, log_value

from newmodel import Generator, Discriminator, FeatureExtractor
from utilsnew import Visualizer2

parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', type=str, default='cifar100', help='cifar10 | cifar100 | folder')
parser.add_argument('--dataroot', type=str, default='../imagegen/maskdata', help='path to dataset')
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--batchsize', type=int, default=16, help='input batch size')
parser.add_argument('--imagesize', type=int, default=100, help='the low resolution image size')
parser.add_argument('--upSampling', type=int, default=2, help='low to high resolution scaling factor')
parser.add_argument('--nEpochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--gEpochs', type=int, default=5, help='number of epochs to pre-train the generator for')
parser.add_argument('--lrG', type=float, default=0.00001, help='learning rate for generator')
parser.add_argument('--lrD', type=float, default=0.0000001, help='learning rate for discriminator')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
# parser.add_argument('--nGPU', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', type=str, default='', help="path to netG (to continue training)")
parser.add_argument('--netD', type=str, default='', help="path to netD (to continue training)")
parser.add_argument('--out', type=str, default='checkpoints', help='folder to output model checkpoints')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.out)
except OSError:
    pass

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

transform = transforms.Compose([transforms.Resize((opt.imagesize*opt.upSampling,opt.imagesize*opt.upSampling)), 
                                transforms.ToTensor()]) #opt.upSampling

transformmask = transforms.Compose([transforms.Resize((2*opt.imagesize*opt.upSampling,opt.imagesize*opt.upSampling)), 
                                transforms.ToTensor()]) #opt.upSampling

normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                std = [0.229, 0.224, 0.225])

scaleandnorm = transforms.Compose([transforms.ToPILImage(),
                            transforms.Resize(opt.imagesize),
                            transforms.ToTensor(),
                            transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                std = [0.229, 0.224, 0.225])
                            ])

scale = transforms.Compose([transforms.ToPILImage(),
                            transforms.Resize(opt.imagesize),
                            transforms.ToTensor()
                            ])


backtrans= transforms.Compose([transforms.Normalize(mean = [-2.118, -2.036, -1.804], #Equivalent to un-normalizing ImageNet (for correct visualization)
                            std = [4.367, 4.464, 4.444]),
                            transforms.ToPILImage(),
                            transforms.Resize(opt.imagesize)])

dataset = datasets.ImageFolder(root= os.path.join(opt.dataroot, 'fakeo') ,
                                transform=transformmask)
datasetreal = datasets.ImageFolder(root= os.path.join(opt.dataroot, 'realo') ,
                                transform=transform)
# assert dataset

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchsize,
                                         shuffle=True, num_workers=int(opt.workers))


dataloaderreal = torch.utils.data.DataLoader(datasetreal, batch_size=opt.batchsize,
                                         shuffle=True, num_workers=int(opt.workers))


netG = Generator(16, opt.upSampling) 
netD = Discriminator()


# For the content loss
feature_extractor = FeatureExtractor(torchvision.models.vgg19(pretrained=True))
print feature_extractor
content_criterion = nn.MSELoss()
adversarial_criterion = nn.BCELoss()

target_real = Variable(torch.ones(opt.batchsize,1))
target_fake = Variable(torch.zeros(opt.batchsize,1))

# if gpu is to be used
if opt.cuda:
    netG.cuda()
    netG = torch.nn.DataParallel(netG)
    netD.cuda()
    netD = torch.nn.DataParallel(netD)
    feature_extractor.cuda()
    content_criterion.cuda()
    adversarial_criterion.cuda()
    target_real = target_real.cuda()
    target_fake = target_fake.cuda()


if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print netG

if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print netD

optimG = optim.Adam(netG.parameters(), lr=opt.lrG)
optimD = optim.SGD(netD.parameters(), lr=opt.lrD, momentum=0.9, nesterov=True)

configure('logs/' + 'genimage-' + str(opt.out) + str(opt.batchsize) + '-' + str(opt.lrG) + '-' + str(opt.lrD), flush_secs=5)
visualizer = Visualizer2()
dire ='resultimages/' +str(opt.out) +'/'
if not os.path.exists(dire):
    os.makedirs(dire)

inputsG = torch.FloatTensor(opt.batchsize, 3, opt.imagesize, opt.imagesize)
inputsGmask = torch.FloatTensor(opt.batchsize, 3, opt.imagesize*opt.upSampling, opt.imagesize*opt.upSampling)
inputsGimg = torch.FloatTensor(opt.batchsize, 3, opt.imagesize*opt.upSampling, opt.imagesize*opt.upSampling)
inputsGreal = torch.FloatTensor(opt.batchsize, 3, opt.imagesize*opt.upSampling, opt.imagesize*opt.upSampling)

siz =opt.imagesize*opt.upSampling
count =0
# Pre-train generator
print 'Generator pre-training'
for epoch in range(opt.gEpochs):
    for i, data in enumerate(dataloader):
        # Generate data
        inputs, _ = data

        if not(int(inputs.size()[0]) == opt.batchsize):
            continue
        # Downsample images to low resolution
        for j in range(opt.batchsize):
            # torchvision.utils.save_image(inputs[j],'main' + str(count) + '.jpg')
            # img= 
            # torchvision.utils.save_image(img,'sample' + str(count) + '.jpg')
            # count+=1
            inputsG[j] = scaleandnorm(inputs[j][:,:siz,:])
            inputsGmask[j] = (1- (inputs[j][:,siz:,:]))
            inputsGimg[j] = normalize(inputs[j][:,:siz,:])
            # torchvision.utils.save_image(inputsGmask[j],'mask' + str(count) + '.jpg')
            # print(inputsGmask[j] )
            
            # while 1:
                # pass

        # Generate real and fake inputs
        if opt.cuda:
            inputsD_real = Variable(inputsGimg.cuda())
            inputmask = Variable(inputsGmask.cuda())
            inputsD_fake = netG(Variable(inputsG).cuda())
        else:
            inputsD_real = Variable(inputsGimg)
            inputmask = Variable(inputsGmask)
            inputsD_fake = netG(Variable(inputsG))

        ######### Train generator #########
        netG.zero_grad()
        lossG_content = content_criterion(inputsD_fake*inputmask, inputsD_real*inputmask)
        lossG_content.backward()

        # Update generator weights
        optimG.step()

        # Status and display
        if i%50==0:
            print('[%d/%d][%d/%d] Loss_G: %.4f'% (epoch, opt.gEpochs, i, len(dataloader), lossG_content.data[0],))
        count= visualizer.show( inputsD_real.cpu().data, inputsD_fake.cpu().data,count,str(opt.out))

    log_value('G_pixel_loss', lossG_content.data[0], epoch)
    torch.save(netG.state_dict(), '%s/netG_pretrain_%d.pth' % (opt.out, epoch))


print 'Adversarial training'
lenreal = len(dataloaderreal)
count=0
visualcount=0
realdata = iter(dataloaderreal)
for epoch in range(opt.nEpochs):
    mean_generator_content_loss = 0.0
    mean_generator_adversarial_loss = 0.0
    mean_generator_total_loss = 0.0
    mean_discriminator_loss = 0.0
    mean_discriminator_realloss = 0.0
    for i, data in enumerate(dataloader):

        ######### Train discriminator #########
        netD.zero_grad()

        # With real data
        if count==lenreal-1:
            del realdata
            del inputsreal
            realdata = iter(dataloaderreal)
        count= (count+1)%lenreal
        inputsreal, _ = realdata.next()
     
        while not(int(inputsreal.size()[0]) == opt.batchsize):
            if count==lenreal-1:
                del realdata
                del inputsreal
                realdata = iter(dataloaderreal)
            count= (count+1)%lenreal
            inputsreal, _ = realdata.next()
        for k in range(opt.batchsize):
            inputsGreal[k] = normalize(inputsreal[k])

        if opt.cuda:
            inputsDreal = Variable(inputsGreal.cuda())
        else:
            inputsDreal = Variable(inputsGreal)

        outputsre = netD(inputsDreal)
        Dreal = outputsre.data.mean()
        lossDreal = adversarial_criterion(outputsre, target_real)
        mean_discriminator_realloss+=lossDreal.data[0]

        # fake data
        inputs, _ = data

        if not(int(inputs.size()[0]) == opt.batchsize):
            continue

        for j in range(opt.batchsize):
            inputsG[j] = scaleandnorm(inputs[j][:,:siz,:])
            inputsGmask[j] = inputs[j][:,siz:,:]/255
            inputs[j] = normalize(inputs[j][:,:siz,:])

        # Generate real and fake inputs
        if opt.cuda:
            inputsD_real = Variable(inputs.cuda())
            inputsD_fake = netG(Variable(inputsG).cuda())
        else:
            inputsD_real = Variable(inputs)
            inputsD_fake = netG(Variable(inputsG))

        outputs = netD(inputsD_real)
        D_real = outputs.data.mean()

        # lossD_real = adversarial_criterion(outputs, target_fake)
        # lossD_real.backward()

        outputsnew = netD(inputsD_fake.detach()) # Don't need to compute gradients wrt weights of netG (for efficiency)
        D_fake = outputsnew.data.mean()

        if i%50==0:
            lossD = adversarial_criterion(outputsnew, target_fake) + 10*(adversarial_criterion(outputs, target_fake) + lossDreal)
        else:
            lossD = 10*(adversarial_criterion(outputs, target_fake) + lossDreal)

        mean_discriminator_loss+=lossD.data[0]
        lossD.backward()

        # Update discriminator weights
        optimD.step()

        ######### Train generator #########
        netG.zero_grad()

        real_features = Variable(feature_extractor(inputsD_real).data)
        fake_features = feature_extractor(inputsD_fake)

        lossG_content = content_criterion(fake_features, real_features)
        lossG_adversarial = adversarial_criterion(netD(inputsD_fake), target_fake)
        mean_generator_content_loss += lossG_content.data[0]

        lossG_total = 0.1*lossG_content + lossG_adversarial 
        mean_generator_adversarial_loss += lossG_adversarial.data[0]
        
        mean_generator_total_loss += lossG_total.data[0]
        lossG_total.backward()

        # Update generator weights
        optimG.step()

        # Status and display
        if i%50==0:
            print('[%d/%d][%d/%d] Dreal(x): %.4f D(x): %.4f D(G(z)): %.4f '% (epoch, opt.nEpochs, i, len(dataloader), Dreal, D_real, D_fake ))
            print('[%d/%d][%d/%d] LossDtotal: %.4f Loss_G (Content/Advers): %.4f/%.4f  Loss_Dreal: %.4f Loss_Dfake: %.4f'
                  % (epoch, opt.nEpochs, i, len(dataloader),lossD.data[0], lossG_content.data[0],
                     lossG_adversarial.data[0], lossDreal.data[0], lossD.data[0]-lossDreal.data[0]))
        if i%200==0:
            visualcount = visualizer.show(inputsG, inputsD_fake.cpu().data,visualcount,str(opt.out))
    log_value('D_real_loss', mean_discriminator_realloss/len(dataloader), epoch)
    log_value('D_fake_loss',(mean_discriminator_loss-mean_discriminator_realloss)/len(dataloader), epoch)
    log_value('G_content_loss', mean_generator_content_loss/len(dataloader), epoch)
    log_value('G_advers_loss', mean_generator_adversarial_loss/len(dataloader), epoch)
    log_value('generator_total_loss', mean_generator_total_loss/len(dataloader), epoch)
    log_value('D_advers_loss', mean_discriminator_loss/len(dataloader), epoch)

    # Do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.out, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.out, epoch))

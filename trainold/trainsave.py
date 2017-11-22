import matplotlib
# import matplotlib.pyplot as plt
matplotlib.use('Agg')

import argparse
import os

import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable

import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from tensorboard_logger import configure, log_value

from models import Generator, Discriminator, FeatureExtractor
from utilsnew import Visualizer2

parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', type=str, default='cifar100', help='cifar10 | cifar100 | folder')
parser.add_argument('--dataroot', type=str, default='./data', help='path to dataset')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=16, help='input batch size')
parser.add_argument('--imageSize', type=int, default=100, help='the low resolution image size')
parser.add_argument('--upSampling', type=int, default=2, help='low to high resolution scaling factor')
parser.add_argument('--nEpochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--gEpochs', type=int, default=50, help='number of epochs to pre-train the generator for')
parser.add_argument('--lrG', type=float, default=0.00001, help='learning rate for generator')
parser.add_argument('--lrD', type=float, default=0.0000001, help='learning rate for discriminator')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--nGPU', type=int, default=1, help='number of GPUs to use')
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

transform = transforms.Compose([transforms.Resize((opt.imageSize*opt.upSampling,opt.imageSize*opt.upSampling)), 
                                transforms.ToTensor()]) #opt.upSampling

normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                std = [0.229, 0.224, 0.225])

scale = transforms.Compose([transforms.ToPILImage(),
                            transforms.Resize(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                std = [0.229, 0.224, 0.225])
                            ])


backtrans= transforms.Compose([transforms.Normalize(mean = [-2.118, -2.036, -1.804], #Equivalent to un-normalizing ImageNet (for correct visualization)
                            std = [4.367, 4.464, 4.444]),
                            transforms.ToPILImage(),
                            transforms.Resize(opt.imageSize)])

dataset = datasets.ImageFolder(root= os.path.join(opt.dataroot, 'fake') ,
                                transform=transform)
datasetreal = datasets.ImageFolder(root= os.path.join(opt.dataroot, 'real') ,
                                transform=transform)
# assert dataset

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))


dataloaderreal = torch.utils.data.DataLoader(datasetreal, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

netG = Generator(3, opt.upSampling-1) #6
#netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print netG

netD = Discriminator()
#netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print netD

# For the content loss
feature_extractor = FeatureExtractor(torchvision.models.vgg19(pretrained=True))
print feature_extractor
content_criterion = nn.MSELoss()
adversarial_criterion = nn.BCELoss()

target_real = Variable(torch.ones(opt.batchSize,1))
target_fake = Variable(torch.zeros(opt.batchSize,1))

# if gpu is to be used
if opt.cuda:
    netG.cuda()
    netD.cuda()
    feature_extractor.cuda()
    content_criterion.cuda()
    adversarial_criterion.cuda()
    target_real = target_real.cuda()
    target_fake = target_fake.cuda()

optimG = optim.Adam(netG.parameters(), lr=opt.lrG)
optimD = optim.SGD(netD.parameters(), lr=opt.lrD, momentum=0.9, nesterov=True)

configure('logs/' + 'genimage-' + str(opt.out) + str(opt.batchSize) + '-' + str(opt.lrG) + '-' + str(opt.lrD), flush_secs=5)
visualizer = Visualizer2()

dire ='resultimages/' +str(opt.out) +'/'
if not os.path.exists(dire):
    os.makedirs(dire)

inputsG = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)

inputsGreal = torch.FloatTensor(opt.batchSize, 3, opt.imageSize*opt.upSampling, opt.imageSize*opt.upSampling)

# Pre-train generator
print 'Generator pre-training'
for epoch in range(opt.gEpochs):
    for i, data in enumerate(dataloader):
        # Generate data
        inputs, _ = data

        # print(int(inputs.size()[0]) )
        if not(int(inputs.size()[0]) == opt.batchSize):
            continue
        # print(inputs.size())
        # Downsample images to low resolution
        for j in range(opt.batchSize):
            inputsG[j] = scale(inputs[j])
            inputs[j] = normalize(inputs[j])



        # print(inputsG.size())
        # print(inputs.size())
        # Generate real and fake inputs
        if opt.cuda:
            inputsD_real = Variable(inputs.cuda())
            inputsD_fake = netG(Variable(inputsG).cuda())
        else:
            inputsD_real = Variable(inputs)
            inputsD_fake = netG(Variable(inputsG))


        # imgplot = plt.imshow(backtrans(inputs[0]))
        # plt.show()

        ######### Train generator #########
        netG.zero_grad()
        # print(inputsD_fake.size())
        # print(inputsD_real.size())

        lossG_content = content_criterion(inputsD_fake, inputsD_real)
        lossG_content.backward()

        # Update generator weights
        optimG.step()

        if i%50==0:
            # Status and display
            print('[%d/%d][%d/%d] Loss_G: %.4f'
                  % (epoch, 50, i, len(dataloader), lossG_content.data[0],))
                # imgplot = plt.imshow(backtrans(inputs[0]))
                # imgplot = plt.imshow(backtrans(inputsD_real.cpu().data[0]))
                # imgplot = plt.imshow(backtrans(inputsD_fake.cpu().data[0]))
                # plt.show()
        # visualizer.show(inputsG, inputsD_real.cpu().data, inputsD_fake.cpu().data)

    log_value('G_pixel_loss', lossG_content.data[0], epoch)
    torch.save(netG.state_dict(), '%s/netG_pretrain_%d.pth' % (opt.out, epoch))

count=0
print 'Adversarial training'
for epoch in range(opt.nEpochs):
    for i, data in enumerate(dataloader):
        # Generate data
        inputs, _ = data

        if not(int(inputs.size()[0]) == opt.batchSize):
            continue

        # Downsample images to low resolution
        for j in range(opt.batchSize):
            inputsG[j] = scale(inputs[j])
            # print(inputs[j].size())
            # print 'uu'
            inputs[j] = normalize(inputs[j])



        # Generate real and fake inputs
        if opt.cuda:
            inputsD_real = Variable(inputs.cuda())
            inputsD_fake = netG(Variable(inputsG).cuda())
        else:
            inputsD_real = Variable(inputs)
            inputsD_fake = netG(Variable(inputsG))


        ######### Train discriminator #########
        netD.zero_grad()
        # outputs = netD(inputsD_real)
        # With real data
        if i%5==0:
            for j, realdata in enumerate(dataloaderreal):
                inputsreal, _ = realdata
                if not(int(inputsreal.size()[0]) == opt.batchSize):
                    continue
                for k in range(opt.batchSize):
                    # print (inputsreal[k].size())
                    inputsGreal[k] = normalize(inputsreal[k])

                if opt.cuda:
                    inputsDreal = Variable(inputsGreal.cuda())
                else:
                    inputsDreal = Variable(inputsGreal)

                # print(inputsDreal.size())
                outputs = netD(inputsDreal)
                D_real = outputs.data.mean()

                lossD_real = adversarial_criterion(outputs, target_real)
                lossD_real.backward()
                break
            

        # print (inputsD_real.size())
        # With fake data

        outputs = netD(inputsD_real.detach())
        D_real = outputs.data.mean()

        lossD_real = adversarial_criterion(outputs, target_fake)
        lossD_real.backward()

        outputs = netD(inputsD_fake.detach()) # Don't need to compute gradients wrt weights of netG (for efficiency)
        D_fake = outputs.data.mean()

        lossD_fake = adversarial_criterion(outputs, target_fake)
        lossD_fake.backward()

        # Update discriminator weights
        optimD.step()

        ######### Train generator #########
        netG.zero_grad()

        real_features = Variable(feature_extractor(inputsD_real).data)
        fake_features = feature_extractor(inputsD_fake)

        lossG_content = content_criterion(fake_features, real_features)
        lossG_adversarial = adversarial_criterion(netD(inputsD_fake).detach(), target_fake)

        # lossG_total = 0.006*lossG_content + 1e-3*lossG_adversarial # initial loss
        lossG_total = 0.000*lossG_content + lossG_adversarial
        lossG_total.backward()

        # Update generator weights
        optimG.step()
        if i%100==0:
            # Status and display
            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G (Content/Advers): %.4f/%.4f D(x): %.4f D(G(z)): %.4f'
                  % (epoch, opt.nEpochs, i, len(dataloader),
                     (lossD_real + lossD_fake).data[0], lossG_content.data[0], lossG_adversarial.data[0], D_real, D_fake,))
        # if i%5==0:
        # visualizer.show(inputsG, inputsDreal.cpu().data, inputsD_fake.cpu().data)
        
        count=visualizer.show(inputsG, inputsD_fake.cpu().data,count)

    log_value('G_content_loss', lossG_content.data[0], epoch)
    log_value('G_advers_loss', lossG_adversarial.data[0], epoch)
    log_value('D_advers_loss', (lossD_real + lossD_fake).data[0], epoch)

    # Do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.out, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.out, epoch))

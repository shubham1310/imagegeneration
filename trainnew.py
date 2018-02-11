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

from newmodel import Generator, Discriminator, FeatureExtractor #, patchDiscriminator
from utilsnew import Visualizer2, SingleImage

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=str, default='../Main/Img/', help='path to dataset')
parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
parser.add_argument('--type', type=str, default='male', help='male/female')
parser.add_argument('--batchsize', type=int, default=16, help='input batch size')
parser.add_argument('--imagesize', type=int, default=200, help='the low resolution image size')
parser.add_argument('--upSampling', type=int, default=1, help='low to high resolution scaling factor')
parser.add_argument('--nEpochs', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--gEpochs', type=int, default=2, help='number of epochs to pre-train the generator for')
parser.add_argument('--lrG', type=float, default=0.0001, help='learning rate for generator')
parser.add_argument('--lrD', type=float, default=0.0001, help='learning rate for discriminator')
# parser.add_argument('--lrDp', type=float, default=0.0001, help='learning rate for patch discriminator')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--netG', type=str, default='', help="path to netG (to continue training)")
parser.add_argument('--netD', type=str, default='', help="path to netD (to continue training)")
# parser.add_argument('--netDp', type=str, default='', help="path to netDp (to continue training)")
parser.add_argument('--out', type=str, default='checkpoints', help='folder to output model checkpoints')
# parser.add_argument('--patchH', type=int, default=25, help='patch height')
# parser.add_argument('--patchW', type=int, default=25, help='patch width')
parser.add_argument('--disstep', type=int, default=1, help='patch width')
parser.add_argument('--losfac', type=float, default=1.0, help='factor to multiply the content loss of Generator')

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

if opt.type=='male':
    datasetfake = SingleImage(imageFolder= os.path.join(opt.dataroot, 'gen') ,
                                    transform=transformmask)
    datasetreal = SingleImage(imageFolder= os.path.join(opt.dataroot, 'origmale') ,
                                    transform=transform)
else:
    datasetfake = SingleImage(imageFolder= os.path.join(opt.dataroot, 'fegen') ,
                                    transform=transformmask)
    datasetreal = SingleImage(imageFolder= os.path.join(opt.dataroot, 'origfemale') ,
                                    transform=transform)

dataloaderfake = torch.utils.data.DataLoader(datasetfake, batch_size=opt.batchsize,
                                         shuffle=True, num_workers=int(opt.workers))


dataloaderreal = torch.utils.data.DataLoader(datasetreal, batch_size=opt.batchsize,
                                         shuffle=True, num_workers=int(opt.workers))


netG = Generator(6, opt.upSampling) 
netD = Discriminator()
# netDp = patchDiscriminator(opt.patchH,opt.patchW)

# For the content loss
# feature_extractor = FeatureExtractor(torchvision.models.vgg19(pretrained=True))
# print feature_extractor
content_criterion = nn.MSELoss()
adversarial_criterion = nn.BCELoss()

target_real = Variable(torch.ones(opt.batchsize,1))
target_fake = Variable(torch.zeros(opt.batchsize,1))
# target_realpatch = Variable(torch.ones(opt.batchsize,opt.patchW*opt.patchH))
# target_fakepatch = Variable(torch.zeros(opt.batchsize,opt.patchW*opt.patchH))

# if gpu is to be used
if opt.cuda:
    netG.cuda()
    netG = torch.nn.DataParallel(netG)
    netD.cuda()
    netD = torch.nn.DataParallel(netD)
    content_criterion.cuda()
    adversarial_criterion.cuda()
    target_real = target_real.cuda()
    target_fake = target_fake.cuda()
    # target_fakepatch = target_fakepatch.cuda()
    # target_realpatch = target_realpatch.cuda()
    # netDp.cuda()
    # netDp = torch.nn.DataParallel(netDp)
    # feature_extractor.cuda()


if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print netG

if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print netD

# if opt.netDp != '':
#     netDp.load_state_dict(torch.load(opt.netDp))
# print netDp

optimG = optim.Adam(netG.parameters(), lr=opt.lrG)
optimD = optim.SGD(netD.parameters(), lr=opt.lrD, momentum=0.9, nesterov=True)
# optimDp = optim.SGD(netDp.parameters(), lr=opt.lrDp, momentum=0.9, nesterov=True)

configure('logs/' + 'patchimage-' + str(opt.out) + str(opt.batchsize) + '-' + str(opt.lrG) + '-' + str(opt.lrD), flush_secs=5)
visualizer = Visualizer2()
dire ='resultimages/' +str(opt.out) +'/'
if not os.path.exists(dire):
    os.makedirs(dire)

inputsG = torch.FloatTensor(opt.batchsize, 3, opt.imagesize, opt.imagesize)
# inputsGmask = torch.FloatTensor(opt.batchsize, 3, opt.imagesize*opt.upSampling, opt.imagesize*opt.upSampling)
inputsGimg = torch.FloatTensor(opt.batchsize, 3, opt.imagesize*opt.upSampling, opt.imagesize*opt.upSampling)
inputsGreal = torch.FloatTensor(opt.batchsize, 3, opt.imagesize*opt.upSampling, opt.imagesize*opt.upSampling)

count =0
# Pre-train generator
print 'Generator pre-training'
for epoch in range(opt.gEpochs):
    for i, data in enumerate(dataloaderreal):
        # Generate data
        inputs = data

        if not(int(inputs.size()[0]) == opt.batchsize):
            continue
        # Downsample images to low resolution
        for j in range(opt.batchsize):
            inputsG[j] = scaleandnorm(inputs[j])
            inputsGimg[j] = normalize(inputs[j])

        # Generate real and fake inputs
        if opt.cuda:
            orig_imag = Variable(inputsGimg.cuda())
            outputG = netG(Variable(inputsG).cuda())
        else:
            orig_imag = Variable(inputsGimg)
            outputG = netG(Variable(inputsG))

        ######### Train generator #########
        netG.zero_grad()
        lossG_content = content_criterion(outputG, orig_imag)
        lossG_content.backward()

        # Update generator weights
        optimG.step()

        # Status and display
        if i%50==0:
            print('[%d/%d][%d/%d] Loss_G: %.4f'% (epoch, opt.gEpochs, i, len(dataloaderreal), lossG_content.data[0],))
            count= visualizer.show(orig_imag.cpu().data, outputG.cpu().data, count, str(opt.out))

    log_value('G_pixel_loss', lossG_content.data[0], epoch)
    torch.save(netG.state_dict(), '%s/netG_pretrain_%d.pth' % (opt.out, epoch))


print 'Adversarial training'
lenreal = len(dataloaderreal)
count=0
logcount=0
visualcount=0
val = 0
val1m=0
val2m=0
valcount=0
realdata = iter(dataloaderreal)
for epoch in range(opt.nEpochs):
    gcount =0
    dcount=0
    mean_generator_content_loss = 0.0
    mean_generator_adversarial_loss = 0.0
    mean_generator_total_loss = 0.0
    mean_discriminator_loss = 0.0
    mean_discriminator_realloss = 0.0
    for i, data in enumerate(dataloaderfake):
        # fake data
        inputs = data

        if not(int(inputs.size()[0]) == opt.batchsize):
            continue

        for j in range(opt.batchsize):
            inputsG[j] = scaleandnorm(inputs[j])
            inputsGimg[j] = normalize(inputs[j])

        # Generate real and fake inputs
        if opt.cuda:
            orig_imag = Variable(inputsGimg.cuda())
            outputG = netG(Variable(inputsG).cuda())
        else:
            orig_imag = Variable(inputsGimg)
            outputG = netG(Variable(inputsG))


        gcount+=1
        ######### Train generator #########
        netG.zero_grad()

        # real_features = Variable(feature_extractor(inputsD_real*inputmask).data)
        # fake_features = feature_extractor(inputsD_fake*inputmask)
        # lossG_content = content_criterion(fake_features, real_features)
        lossG_adversarial = adversarial_criterion(netD(outputG), target_real) 
        lossG_content = content_criterion(outputG, orig_imag)
        # lossG_content.backward(retain_graph=True)
        # val1 = torch.sum(torch.abs(netG.module.conv3.weight.grad.data))
        # valcount+=1
        # netG.zero_grad()
        # lossG_adversarial = adversarial_criterion(netDp(inputsD_fake), target_realpatch)

        # lossG_adversarial.backward(retain_graph=True)
        # val2 = torch.sum(torch.abs(netG.module.conv3.weight.grad.data))
        # print('G_content: %.4f G_adversarial: %.4f G_content/G_adversarial: %.4f'%(val1,val2,val1/val2))
        # val +=val1/val2
        # val1m+=val1
        # val2m+=val2
        # log_value('G_content', val1, valcount)
        # log_value('G_adversarial', val2, valcount)
        # log_value('G_content/G_adversarial', val1/val2, valcount)

        # netG.zero_grad()
        lossG_total = opt.losfac*lossG_content + lossG_adversarial 


        mean_generator_adversarial_loss += lossG_adversarial.data[0]/opt.batchsize
        mean_generator_content_loss += lossG_content.data[0]/opt.batchsize
        mean_generator_total_loss += lossG_total.data[0]/opt.batchsize


        lossG_total.backward()
        # grad_of_param = {}
            # for name, parameter in netG.named_parameters():
            #     grad_of_param[name] = parameter.grad
            #     print(name)


        # Update generator weights
        optimG.step()

        if i%opt.disstep==0:
            dcount+=1
            ######### Train discriminator #########
            netD.zero_grad()
            # netDp.zero_grad()

            # With real data
            if count==lenreal-1:
                del realdata
                del inputsreal
                realdata = iter(dataloaderreal)
            count= (count+1)%lenreal
            inputsreal = realdata.next()
         
            while not(int(inputsreal.size()[0]) == opt.batchsize):
                if count==lenreal-1:
                    del realdata
                    del inputsreal
                    realdata = iter(dataloaderreal)
                count= (count+1)%lenreal
                inputsreal = realdata.next()
            for k in range(opt.batchsize):
                inputsGreal[k] = normalize(inputsreal[k])

            if opt.cuda:
                inputsDreal = Variable(inputsGreal.cuda())
            else:
                inputsDreal = Variable(inputsGreal)

            outputsre = netD(inputsDreal)
            Dreal = outputsre.data.mean()
            lossDreal = adversarial_criterion(outputsre, target_real) 

            # Dreal = outputsrepatch.data.mean()
            # outputsrepatch = netDp(inputsDreal)
            # lossDreal = adversarial_criterion(outputsrepatch,target_realpatch)


            outputsnew = netD(outputG.detach()) # Don't need to compute gradients wrt weights of netG (for efficiency)
            D_fake = outputsnew.data.mean()
            lossD =lossDreal + adversarial_criterion(outputsnew, target_fake) 

            # outputsnewpatch = netDp(inputsD_fake.detach())
            # lossD =lossDreal + adversarial_criterion(outputsnewpatch,target_fakepatch) 


            mean_discriminator_realloss+=lossDreal.data[0]/opt.batchsize
            mean_discriminator_loss+=lossD.data[0]/opt.batchsize
            lossD.backward()
            
            # Update discriminator weights
            optimD.step()
            # optimDp.step()


        # Status and display
        if i%50==0:
            # print("Average G_content: %.4f G_adversarial: %.4f  G_content/G_adversarial: %.4f"%(val1m/valcount,val2m/valcount,val/valcount))
            print('[%d/%d][%d/%d] Dreal(x): %.4f D(G(z)): %.4f '% (epoch, opt.nEpochs, i, len(dataloaderreal), Dreal, D_fake ))
            print('[%d/%d][%d/%d] Loss_G (Content/Advers): %.4f/%.4f  Loss_Dreal: %.4f Loss_Dfake: %.4f LossDtotal: %.4f  LossGtotal: %.4f'
                  % (epoch, opt.nEpochs, i, len(dataloaderreal), lossG_content.data[0], lossG_adversarial.data[0],
                      lossDreal.data[0], lossD.data[0]-lossDreal.data[0], lossD.data[0],lossG_total.data[0]))
        if i%200==0:
            visualcount = visualizer.show(inputsG, outputG.cpu().data,visualcount,str(opt.out))
            log_value('D_realloss', mean_discriminator_realloss/dcount, logcount)
            log_value('D_fakeloss',(mean_discriminator_loss-mean_discriminator_realloss)/dcount, logcount)
            log_value('D_totalloss', mean_discriminator_loss/dcount, logcount)
            log_value('G_contentloss', mean_generator_content_loss/gcount, logcount)
            log_value('G_adversloss', mean_generator_adversarial_loss/gcount, logcount)
            log_value('G_totalloss', mean_generator_total_loss/gcount, logcount)
            mean_generator_content_loss = 0.0
            mean_generator_adversarial_loss = 0.0
            mean_generator_total_loss = 0.0
            mean_discriminator_loss = 0.0
            mean_discriminator_realloss = 0.0
            gcount =0
            dcount=0
            logcount+=1

    # Do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.out, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.out, epoch))
    # torch.save(netDp.state_dict(), '%s/netDp_epoch_%d.pth' % (opt.out, epoch))

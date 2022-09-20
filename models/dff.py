import torch.nn as nn
import torch
import os,sys
sys.path.append('.')
from models.resnet_ori import resnet101
import torch.nn.functional as F

class DFF(nn.Module):
    def __init__(self,nclass = 19,pretrained = True,norm_layer=nn.BatchNorm2d,device = "cuda"):
        super().__init__()
        self.nclass = nclass
        self.pretrained = resnet101(pretrained = pretrained)

        self.EW1 = LocationAdaptiveLearner(nclass, nclass*4, nclass*4)

        self.side1 = nn.Sequential(nn.Conv2d(64, 1, 1),
                                   norm_layer(1))
        self.side2 = nn.Sequential(nn.Conv2d(256, 1, 1, bias=True),
                                   norm_layer(1),
                                   nn.ConvTranspose2d(1, 1, 4, stride=2, padding=1, bias=False))
        self.side3 = nn.Sequential(nn.Conv2d(512, 1, 1, bias=True),
                                   norm_layer(1),
                                   nn.ConvTranspose2d(1, 1, 8, stride=4, padding=2, bias=False))
        self.side5 = nn.Sequential(nn.Conv2d(2048, nclass, 1, bias=True),
                                   norm_layer(nclass),
                                   nn.ConvTranspose2d(nclass, nclass, 16, stride=8, padding=4, bias=False))

        self.side5_ew = nn.Sequential(nn.Conv2d(2048, nclass*4, 1, bias=True),
                                   norm_layer(nclass*4),
                                   nn.ConvTranspose2d(nclass*4, nclass*4, 16, stride=8, padding=4, bias=False))
        

        # transform to specified device
        self.pretrained = self.pretrained.to(device)
        self.EW1 = self.EW1.to(device)
        self.side1 = self.side1.to(device)
        self.side2 = self.side2.to(device)
        self.side3 = self.side3.to(device)
        self.side5 = self.side5.to(device)
        self.side5_ew = self.side5_ew.to(device)



    def forward(self,x):
        f1,f2,f3,f4,f5 = self.pretrained(x) # /1,
        side1 = self.side1(f1) # (N, 1, H, W)
        side2 = self.side2(f2) # (N, 1, H, W)
        side3 = self.side3(f3) # (N, 1, H, W)
        side5 = self.side5(f5) # (N, 19, H, W)
        side5_w = self.side5_ew(f5) # (N, 19*4, H, W)

        ada_weights = self.EW1(side5_w) # (N, 19, 4, H, W)
        slice5 = side5[:,0:1,:,:] # (N, 1, H, W)
        fuse = torch.cat((slice5, side1, side2, side3), dim = 1)
        for i in range(side5.size(1)-1):
            slice5 = side5[:,i+1:i+2,:,:] # (N, 1, H, W)
            fuse = torch.cat((fuse, slice5, side1, side2, side3), dim = 1) # (N, 19*4, H, W)

        fuse = fuse.view(fuse.size(0), self.nclass, -1, fuse.size(2), fuse.size(3)) # (N, 19, 4, H, W)
        fuse = torch.mul(fuse, ada_weights) # (N, 19, 4, H, W)
        fuse = torch.sum(fuse, dim = 2) # (N, 19, H, W)
        

        outputs = [side5, fuse]

        return tuple(outputs)

class LocationAdaptiveLearner(nn.Module):
    """docstring for LocationAdaptiveLearner"""
    def __init__(self, nclass, in_channels, out_channels, norm_layer=nn.BatchNorm2d):
        super(LocationAdaptiveLearner, self).__init__()
        self.nclass = nclass

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=True),
                                   norm_layer(out_channels),
                                   nn.ReLU(inplace=True))
        self.conv2 = nn.Sequential(nn.Conv2d(out_channels, out_channels, 1, bias=True),
                                   norm_layer(out_channels),
                                   nn.ReLU(inplace=True))
        self.conv3 = nn.Sequential(nn.Conv2d(out_channels, out_channels, 1, bias=True),
                                   norm_layer(out_channels))

    def forward(self, x):
        # x:side5_w (N, 19*4, H, W)
        x = self.conv1(x) # (N, 19*4, H, W)
        x = self.conv2(x) # (N, 19*4, H, W)
        x = self.conv3(x) # (N, 19*4, H, W)
        x = x.view(x.size(0), self.nclass, -1, x.size(2), x.size(3)) # (N, 19, 4, H, W)
        return x



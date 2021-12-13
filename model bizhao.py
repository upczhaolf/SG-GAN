import math
import torch.nn.functional as F
from torch import nn
import torch

class DenseBlock(nn.Module):
    def __init__(self, channels):
        super(DenseBlock, self).__init__()

        self.relu = nn.PRelu()
        self.conv1 = nn.Conv2d(in_channels= channels, out_channels= 16, kernel_size= 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size = 3, stride=1,padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size = 3, stride=1,padding=1 )
        self.conv4 = nn.Conv2d(in_channels=48, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=80, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(in_channels=96, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(in_channels=112, out_channels=16, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        conv1 = self.relu(self.conv(x))

        conv2= self.relu(self.conv2(conv1))
        #cout2_dense = self.relu(torch.cat([conv1,conv2],1))

        conv3=self.relu(self.conv3(conv2))
        #cout3_dense = self.relu(torch.cat([conv1,conv2,conv3],1))

        conv4 = self.relu(self.conv4(conv3))
        #cout4_dense=self.relu(torch.cat([conv1,conv2,conv3,conv4],1))

        conv5=self.relu(self.conv5(conv4))
        #cout5_dense=self.relu(torch.cat([conv1,conv2,conv3,conv4,conv5],1))

        conv6 = self.relu(self.conv6(conv5))
        #cout6_dense = self.relu(torch.cat([conv1,conv2,conv3,conv4,conv5,conv6],1))

        conv7 = self.relu(self.conv7(conv6))
        #cout7_dense = self.relu(torch.cat([conv1,conv2,conv3,conv4,conv5,conv6,conv7],1))

        conv8 = self.relu(self.conv8(conv7))
        cout_dense = self.relu(torch.cat([conv1,conv2,conv3,conv4,conv5,conv6,conv7,conv8],1))

        return cout_dense

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.relu = nn.PReLU()
        self.bottleneck = nn.Conv2d(in_channels = 896, out_channels =256, kernel_size=1, stride=1, padding=0)
        self.reconstruction = nn.Conv2d(in_channels=256, out_channels =3, kernel_size=3, stride=1,padding=1,bias=False)
        self.lowlevel = nn.Conv2d(in_channels= 1,out_channels= 128, kernel_size= 3,stride=1,padding=1,bias=False)
        self.block1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=9, padding=4),
                                    nn.PReLU())
        self.denseblock1 = self.make_layer(DenseBlock, 128)
        self.denseblock2 = self.make_layer(DenseBlock, 256)
        self.denseblock3 = self.make_layer(DenseBlock, 384)
        self.denseblock4 = self.make_layer(DenseBlock, 512)
        self.denseblock5 = self.make_layer(DenseBlock, 640)
        self.denseblock6 = self.make_layer(DenseBlock, 768)
        #self.denseblock7 = self.make_layer(DenseBlock, 896)
        self.denconv = nn.Sequential(
            nn.ConvTransposed2d(in_channels=256, out_channels=256,kernel_size=2,stride=2,padding=0,bias=False ),
            nn.PReLU(),
            nn.ConvTranspose2d(in_channels=256, out_channels=256,kernel_size=2,stride=2, padding=0,bias=False),
            nn.PReLU(),
        )
    def make_layer(self, block,channel_in):
            layers = []
            layers.append(block(channel_in))
            return nn.Sequential(*layers)
    def forward(self,x):
            residual = self.relu(self.block1)

            out = self.denseblock1(residual)
            concat = torch.cat([residual,out], 1)

            out = self.denseblock2(concat)
            concat = torch.cat([residual,out], 1)

            out = self.denseblock3(concat)
            concat = torch.cat([residual,out], 1)

            out = self.denseblock4(concat)
            concat = torch.cat([residual,out], 1)

            out = self.denseblock5(concat)
            concat = torch.cat([residual,out],1)

            out = self.denseblock6(concat)
            out = torch.cat([residual,out], 1)

            out = self.bottleneck(out)

            out = self.deconv(out)

            out = self.reconstruction(out)

            return out

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        return F.sigmoid(self.net(x).view(batch_size))

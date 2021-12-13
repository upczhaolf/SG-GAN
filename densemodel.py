import math
import torch.nn.functional as F
import torch
from torch import nn
import numpy

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        #self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        #self.bn2 = nn.BatchNorm2d(channels)


    def forward(self, x):
        residual = self.prelu(self.conv1(x))
        #residual = self.bn1(residual)
        # residual1 = self.prelu(residual)
        residual = self.conv2(residual)
        residual *= 0.1
        #residual = self.bn2(residual)

        residual5 =residual+x

        return  residual5


class Generator(nn.Module):
    def __init__(self, scale_factor):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(Generator, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        #self.residual = self.make_layer(ResidualBlock, 32)
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block9 = ResidualBlock(64)
        self.block10 = ResidualBlock(64)
        self.block11 = ResidualBlock(64)
        self.block12 = ResidualBlock(64)
        self.block13 = ResidualBlock(64)
        self.block14 = ResidualBlock(64)
        self.block15 = ResidualBlock(64)
        self.block16 = ResidualBlock(64)
        self.block17 = ResidualBlock(64)
        self.block18 = ResidualBlock(64)
        self.block19 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 3, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block9 = self.block9(block5)
        block10 = self.block10(block9)
        block11 = self.block11(block10)
        block12 = self.block12(block11)
        block13 = self.block13(block12)
        block14 = self.block14(block13)
        block15 = self.block15(block14)
        block16 = self.block16(block15)
        block17 = self.block17(block16)
        block18 = self.block18(block17)
        block19 = self.block19(block18)

        block7 = self.block7(block19+block1)
        block8 = self.block8(block7)

        return (F.tanh(block8) + 1) / 2

    def make_layer(self,block,num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)


# class Generator(nn.Module):
#     def __init__(self, scale_factor):
#         upsample_block_num = int(math.log(scale_factor, 2))
#
#         super(Generator, self).__init__()
#         self.block1 = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=9, padding=4),
#             nn.PReLU()
#         )
#         self.block2 = ResidualBlock(64)
#         self.block3 = ResidualBlock(64)
#         self.block4 = ResidualBlock(64)
#         self.block5 = ResidualBlock(64)
#         self.block6 = ResidualBlock(64)
#         self.block7 = nn.Sequential(
#             nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64)
#         )
#         block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
#         block8.append(nn.Conv2d(64, 3, kernel_size=9, padding=4))
#         self.block8 = nn.Sequential(*block8)
#
#     def forward(self, x):
#         block1 = self.block1(x)
#         block2 = self.block2(block1)
#         block3 = self.block3(block2)
#         block4 = self.block4(block3)
#         block5 = self.block5(block4)
#         block6 = self.block6(block5)
#         block7 = self.block7(block6)
#         block8 = self.block8(block1 + block7)
#
#         return (F.tanh(block8) + 1) / 2
# class Generator(nn.Module):
#     def __init__(self, upsacle_factor):
#         upsample_block_num=int(math.log(upsacle_factor, 2))
#         super(Generator, self).__init__()
#         # super(Generator, self).init_()
#         self.block1=nn.Sequential(nn.Conv2d(3, 64, kernel_size=9, padding=4),
#                                   nn.PReLU())
#         num_layers =6
#         bn_size = 4
#         in_channels = 64
#         growth_rate=12
#         self.block2 = DenseBlock(num_layers,in_channels, bn_size,growth_rate)
#         # self.block3 = DenseBlock(num_layers,in_channels, bn_size,growth_rate)
#         # self.block5 = DenseBlock()
#         self.block3= Transition(200, 128)
#         block6=[UpsampleBLock(128,2) for _ in range(upsample_block_num)]
#         block6.append(nn.Conv2d(128,3,kernel_size=9, padding=4 ))
#         self.block6 = nn.Sequential(*block6)
#
#     def forward(self, x):
#         block1 = self.block1(x)
#         block2 = self.block2(block1)
#         block3 = self.block3(torch.cat([block1,block2],1))
#         block6 = self.block6(block3)
#
#
#         return (F.tanh(block6)+1)/2

# class DenseBlock(nn.Module):
#     def __init__(self, channels):
#         super(DenseBlock, self).__init__()
#
#         self.relu = nn.PReLU()
#         self.conv1 = nn.Conv2d(in_channels=channels, out_channels=16, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
#         self.conv3 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
#         self.conv4 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
#         self.conv5 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
#         self.conv6 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
#         self.conv7 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
#         self.conv8 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1)
#
#     def forward(self, x):
#         conv1 = self.relu(self.conv1(x))
#
#         conv2 = self.relu(self.conv2(conv1))
#         cout2_dense = self.relu(torch.cat([conv1, conv2], 1))
#
#         #conv3 = self.relu(self.conv3(conv2 ))
#         #cout3_dense = self.relu(torch.cat([conv1, conv2, conv3], 1))
#
#         # conv4 = self.relu(self.conv4(conv3 ))
#         # cout4_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4], 1))
#         #
#         # conv5 = self.relu(self.conv5(conv4 ))
#         # cout5_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4, conv5], 1))
#         #
#         # conv6 = self.relu(self.conv6(conv5 ))
#         # cout6_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4, conv5, conv6], 1))
#         #
#         # conv7 = self.relu(self.conv7(conv6 ))
#         # cout7_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4, conv5, conv6, conv7], 1))
#
#         conv8 = self.relu(self.conv8(conv2))
#         #cout_dense = self.relu(torch.cat([conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8], 1))
#         cout_dense = self.relu(torch.cat([conv8, conv2,conv1],1))
#
#         return cout_dense
#
#
# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__()
#         #upsample_block_num = int(math.log(upsacle_factor, 2))
#         self.relu = nn.PReLU()
#         self.bottleneck = nn.Conv2d(in_channels=416, out_channels=256, kernel_size=1, stride=1, padding=0,bias=False)
#         self.reconstruction = nn.Conv2d(in_channels=256, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
#         self.lowlevel = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=3, stride=1, padding=1)
#         #self.block1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=9, padding=4),
#          #                           nn.PReLU())
#         self.denseblock1 = self.make_layer(DenseBlock, 128)
#         self.denseblock2 = self.make_layer(DenseBlock, 48)
#         self.denseblock3 = self.make_layer(DenseBlock, 48)
#         self.denseblock4 = self.make_layer(DenseBlock, 48)
#         self.denseblock5 = self.make_layer(DenseBlock, 48)
#         self.denseblock6 = self.make_layer(DenseBlock, 48)
#         # self.denseblock7 = self.make_layer(DenseBlock, 896)
#         self.deconv = nn.Sequential(
#             nn.ConvTranspose2d(in_channels=48, out_channels=256, kernel_size=2, stride=2, padding=0, bias=False),
#             nn.PReLU(),
#             nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2, padding=0, bias=False),
#             nn.PReLU()
#         )
#
#         for m in self.modules():
#              if isinstance(m,nn.Conv2d):
#                  n=m.kernel_size[0]*m.kernel_size[1]*m.out_channels
#                  m.weight.data.normal_(0, math.sqrt(2./n))
#                  if m.bias is not None:
#                      m.bias.data.zero_()
#              if isinstance(m, nn.ConvTranspose2d):
#                  c1,c2,h,w = m.weight.data.size()
#                  weight = get_upsample_filter(h)
#                  m.weight.data = weight.view(1,1,h,w).repeat(c1,c2,1,1)
#                  if m.bias is not None:
#                      m.bias.data.zero_()
#
#
#
#
#     def make_layer(self, block, channel_in):
#             layers = []
#             layers.append(block(channel_in))
#             return nn.Sequential(*layers)
#
#     def forward(self, x):
#             residual = self.relu(self.lowlevel(x))
#
#             out = self.denseblock1(residual)
#             #concat1 = torch.cat([residual, out], 1)
#

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


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual
class DenseLayer(nn.Sequential):
    def __init__(self, in_channels,growth_rate, bn_size):
    # def __init__(self, in_channels, growth_rate, bn_size):
        super(DenseLayer,self).__init__()

        self.add_module('norm1',nn.BatchNorm2d(in_channels))
        self.add_module('relu1',nn.ReLU(inplace=True))
        self.add_module('conv1',nn.Conv2d(in_channels, bn_size*growth_rate, kernel_size= 3,
                                          stride=1,padding=1, bias=False))
        self.add_module('norm2',nn.BatchNorm2d(bn_size*growth_rate))
        self.add_module('relu2',nn.ReLU(inplace=True))
        self.add_module('conv2',nn.Conv2d(bn_size*growth_rate, growth_rate, kernel_size= 1,
                                          stride=1, bias=False))


    def forward(self,x):
            new_features = super(DenseLayer,self).forward(x)
            return torch.cat([x,new_features],1)


# class DenseBlock(nn.Sequential):
#     def __init__(self,num_layers,in_channels,bn_size,growth_rate):
#
#         super(DenseBlock , self).__init__()
#         for i in range (num_layers):
#             self.add_module('denselayer%d' % (i+1),
#                             DenseLayer(in_channels+growth_rate*i,
#                                        growth_rate,bn_size))
        # super(DenseBlock, self).__init__()
        # self.bn1 = nn.BatchNorm2d(in_channels)
        # self.prelu = nn.PReLU()
        # self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=1)
        # self.bn2 = nn.BatchNorm2d(out_channels)
        # self.prelu = nn.PReLU()
        # channels = 12
        # self.conv2 = nn.Conv2d(out_channels, channels, kernel_size=3, padding=1)
        #
        # def forward(self, x):
        #     residual = self.bn1(x)
        #     residual = self.prelu(residual)
        #     residual = self.conv1(residual)
        #     residual = self.bn2(residual)
        #     residual = self.prelu(residual)
        #     residual = self.conv2(residual)
        #
        #     return torch.cat([x,residual],1)
        # def forward(self, x):
        #     new_features = super(DenseLayer, self).forward(x)
        #     return torch.cat([x, new_features], 1)



class Transition(nn.Sequential):
    def __init__(self,in_channels,out_channels):
        super(Transition,self).__init__()
        self.add_module('norm',nn.BatchNorm2d(in_channels))
        self.add_module('relu',nn.ReLU(inplace=True))
        self.add_module('conv',nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                         stride= 1, bias=False))
        #self.add_module('pool',nn.AvgPool2d(kernel_size=2, stride=2))






class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x

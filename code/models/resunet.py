import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import resnetv2
import numpy as np
import math

# mx code adapted to pytorch from https://github.com/feevos/resuneta/tree/master
class ResUNet(nn.Module):
    def __init__(self, num_classes=6, depth=6, nFilters=32):
        super().__init__()
        self.num_classes = num_classes
        
        self.depth = depth # from original d6 encoder https://github.com/feevos/resuneta/blob/master/models/resunet_d6_encoder.py
        self.nFilters = nFilters # picked from resunet demo
        
        self.layer_1 = nn.Sequential(*[
            nn.Conv2d(in_channels=3,
                      out_channels=self.nFilters,
                      kernel_size=1,
                      dilation=1,
                      stride=1)
        ])
        
        self.layer_2 = nn.Sequential(*[
            ResBlock(features=self.nFilters, kernel=3, dilation=[1], stride=1)
        ])
        
        self.layer_3_4 = nn.Sequential(*[
            nn.Conv2d(in_channels=self.nFilters * 2 ** 0,
                      out_channels=self.nFilters * 2 ** 1,
                      kernel_size=1,
                      dilation=1,
                      stride=2),
            ResBlock(features=self.nFilters * 2 ** 1, kernel=3, dilation=[1], stride=1)
        ])
        
        self.layer_5_6 = nn.Sequential(*[
            nn.Conv2d(in_channels=self.nFilters * 2 ** 1,
                      out_channels=self.nFilters * 2 ** 2,
                      kernel_size=1,
                      dilation=1,
                      stride=2),
            ResBlock(features=self.nFilters * 2 ** 2, kernel=3, dilation=[1], stride=1)
        ])
        
        
        self.layer_7_8 = nn.Sequential(*[
            nn.Conv2d(in_channels=self.nFilters * 2 ** 2,
                      out_channels=self.nFilters * 2 ** 3,
                      kernel_size=1,
                      dilation=1,
                      stride=2),
            ResBlock(features=self.nFilters * 2 ** 3, kernel=3, dilation=[1], stride=1)
        ])
                
        self.layer_9_10 = nn.Sequential(*[
            nn.Conv2d(in_channels=self.nFilters * 2 ** 3,
                      out_channels=self.nFilters * 2 ** 4,
                      kernel_size=1,
                      dilation=1,
                      stride=2),
            ResBlock(features=self.nFilters * 2 ** 4, kernel=3, dilation=[1], stride=1)
        ])
        
        self.layer_11_12 = nn.Sequential(*[
            nn.Conv2d(in_channels=self.nFilters * 2 ** 4,
                      out_channels=self.nFilters * 2 ** 5,
                      kernel_size=1,
                      dilation=1,
                      stride=2),
            ResBlock(features=self.nFilters * 2 ** 5, kernel=3, dilation=[1], stride=1)
        ])
        
        # We have 5 stride=2 layers and 5 upsampling layers. But psp also has a stride=2, so we need 7
        # upsampling layers... But where does the 6th one go? 
        self.psp_13 = PyramidPooling(self.nFilters * 2 ** 5, nChannels=self.nFilters * 2 ** 5)
        
        self.layer_14 = nn.Sequential(*[
            nn.Upsample(scale_factor=2),
            Downsample(512)
        ])
        
        self.layer_15_16 = nn.Sequential(*[
            Combine(512),
            ResBlock(512)
        ])
        
        self.layer_17 = nn.Sequential(*[
            nn.Upsample(scale_factor=2),
            Downsample(256)
        ])
        
        self.layer_18_19 = nn.Sequential(*[
            Combine(256),
            ResBlock(256)
        ])
        
        self.layer_20 = nn.Sequential(*[
            nn.Upsample(scale_factor=2),
            Downsample(128)
        ])
        
        self.layer_21_22 = nn.Sequential(*[
            Combine(128),
            ResBlock(128)
        ])
        
        self.layer_23 = nn.Sequential(*[
            nn.Upsample(scale_factor=2),
            Downsample(64)
        ])
        
        self.layer_24_25 = nn.Sequential(*[
            Combine(64),
            ResBlock(64)
        ])
        
        self.layer_26 = nn.Sequential(*[
            nn.Upsample(scale_factor=2),
            Downsample(32)
        ])
        
        self.layer_27_28 = nn.Sequential(*[
            Combine(32),
            ResBlock(32)
        ])
        
        self.layer_29 = nn.Sequential(*[
            Combine(32)
            #nn.Upsample(scale_factor=2) # TODO: would I scale here if I reduce the stride of psp?
        ])
        
        self.layer_30_31 = nn.Sequential(*[
            PyramidPooling(32, nChannels=32),
            nn.Conv2d(in_channels=32, out_channels=self.num_classes, kernel_size=1, dilation=1, stride=1)
        ])
        
        self.out = nn.Softmax()
        '''
        PSEUDOCODE
        
        1 layer1 = conv2d, f=32,kernel=1 dialation=1, stride=1
        2 ResBlock = conv2d, f=32, kernel=3 dialation=[1, 3, 15, 31], stride=1
        3 layer1 = conv2d, f=64, kernel=1 dialation=1, stride=2
        4 ResBlock = conv2d, f=64, kernel=3 dialation=[1, 3, 15, 31], stride=1
        5 layer1 = conv2d, f=128, kernel=1 dialation=1, stride=2
        6 ResBlock = conv2d, f=128, kernel=3 dialation=[1, 3, 15], stride=1
        7 layer1 = conv2d, f=256, kernel=1 dialation=1, stride=2
        8 ResBlock = conv2d, f=256, kernel=3 dialation=1, stride=1
        9 layer1 = conv2d, f=512, kernel=1 dialation=1, stride=2
        10 ResBlock = conv2d, f=512, kernel=3 dialation=1, stride=1
        11 layer1 = conv2d, f=1024, kernel=1 dialation=1, stride=2
        12 ResBlock = conv2d, f=1024, kernel=3 dialation=1, stride=1
        13 PSPPooling
        14 Upsample f = 512
        15 Combine f = 512, layers 14 and 10
        16 ResBlock f = 512 k=3
        17 Upsample f = 256
        18 Combine f = 256, layers 17 and 8
        19 ResBlock f = 256 k=3
        20 Upsample f = 128
        21 Combine f = 128, layers 20 and 6
        22 ResBlock f = 128 k=3
        23 Upsample f = 64
        24 Combine f = 64, layers 23 and 4
        25 ResBlock f = 64 k=3
        26 Upsample f = 32
        27 Combine f = 32 layers 26 and 2
        28 ResBlock f=32 k =3
        29 Combine f = 32 layers 28 and 2
        30 PSPPooling
        31 Conv2D f = 6 Layers 28 and 1
        32 Softmax on class dimension
        '''
    def forward(self, x):
        l1 = self.layer_1(x)
        l2 = self.layer_2(l1)
        l3_4 = self.layer_3_4(l2)
        l5_6 = self.layer_5_6(l3_4)
        l7_8 = self.layer_7_8(l5_6)
        l9_10 = self.layer_9_10(l7_8)
        l11_12 = self.layer_11_12(l9_10)
        l13 = self.psp_13(l11_12)
        l14 = self.layer_14(l13)
        l15_16 = self.layer_15_16([l14, l9_10])
        l17 = self.layer_17(l15_16)
        l18_19 = self.layer_18_19([l17, l7_8])
        l20 = self.layer_20(l18_19)
        l21_22 = self.layer_21_22([l20, l5_6])
        l23 = self.layer_23(l21_22)
        l24_25 = self.layer_24_25([l23, l3_4])
        l26 = self.layer_26(l24_25)
        l27_28 = self.layer_27_28([l26, l2])
        l29 = self.layer_29([l27_28, l1])
        l30_31 = self.layer_30_31(l29)
        out = self.out(l30_31)
        
        return out

class Combine(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.features = features
        
        self.relu = nn.ReLU()
        self.conv2d = nn.Conv2d(in_channels=self.features * 2, out_channels=self.features, kernel_size=1, dilation=1, stride=1)
        
    def forward(self, x):
        input1 = x[0]
        input2 = x[1]
        
        input1 = self.relu(input1)
        input2 = torch.cat((input1, input2), dim=1)
        
        return self.conv2d(input2)

class Downsample(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.output_features = features
        self.input_features = features * 2
        
        self.conv2d = nn.Conv2d(in_channels=self.input_features, out_channels=self.output_features,
                                kernel_size=1)
        
    def forward(self, x):
        return self.conv2d(x)
# ResUNet uses Pyramid Pooling https://arxiv.org/pdf/1612.01105.pdf
class PyramidPooling(nn.Module):
    '''
    PSEUDOCODE
    CONV2D features, k=1,d=1,s=2
    ResBlock features, k=3
    MaxPool kernel=2 stride=2
    Upsample
    Concat Upsample and ResBlock
    Conv2D features k=1
    
    I noticed that the code here does not have a stride of 2:
    https://github.com/feevos/resuneta/blob/49d26563f84c737e07d34edfe30b56c59cbb4203/nn/pooling/psp_pooling.py#L8

    It uses a stride of 1 with the normed conv2d:
    https://github.com/feevos/resuneta/blob/49d26563f84c737e07d34edfe30b56c59cbb4203/nn/layers/conv2Dnormed.py#L2
    '''

    def __init__(self, nFeatures, nChannels=3, depth=6):
        super().__init__()
        self.nFeatures = nFeatures
        self.nChannels = nChannels
        self.depth = depth
        
        self.layer_B = nn.Sequential(*[
            nn.Conv2d(in_channels=self.nChannels,
                      out_channels=self.nFeatures,
                      kernel_size=1,
                      dilation=1,
                      stride=1),
            ResBlock(features=self.nFeatures, kernel=3, dilation=[1], stride=1)
        ])
        
        self.layer_D = nn.Sequential(*[
            nn.MaxPool2d(kernel_size=2, stride=2),
            #https://github.com/feevos/resuneta/blob/49d26563f84c737e07d34edfe30b56c59cbb4203/nn/layers/scale.py#L46
            # They want to use Bilinear, but the actual implementation uses nearest bc programming problems. I'll do bilinear
            nn.Upsample(scale_factor=2)
        ])
        
        self.conv_out = nn.Conv2d(in_channels=self.nFeatures * 2,
                                  out_channels=self.nFeatures,
                                  kernel_size=1,
                                  dilation=1,
                                  stride=1)
        
    def forward(self, x):
        layer_b = self.layer_B(x)
        layer_d = self.layer_D(layer_b)
        concat = torch.cat((layer_d, layer_b), dim=1)
        out = self.conv_out(concat)
        return out
        
        
class ResBlock(nn.Module):
    '''
    Image reference for ResBlock: https://www.researchgate.net/figure/Flowchart-of-the-resblock-Each-resblock-is-composed-of-a-batch-normalization-a_fig2_330460151
    '''
    def __init__(self, features, kernel = 3, dilation=[1], stride=1, device="cuda"):
        super().__init__()
        self.f = features
        self.kernel = kernel
        self.dilation = dilation
        self.stride = stride
        self.device = device
        
        normal_sides = [
            nn.BatchNorm2d(self.f),
            nn.ReLU(),
            nn.Conv2d(self.f, self.f, kernel_size=self.kernel, dilation=self.dilation[0], stride=self.stride, padding=self.dilation[0]),
            nn.BatchNorm2d(self.f),
            nn.ReLU(),
            nn.Conv2d(self.f, self.f, kernel_size=self.kernel, dilation=self.dilation[0], stride=self.stride, padding=self.dilation[0])
        ]
        
        for i in range(1, len(dilation)):
            normal_sides = np.vstack((normal_sides, [
            nn.BatchNorm2d(self.f),
            nn.ReLU(),
            nn.Conv2d(self.f, self.f, kernel_size=self.kernel, dilation=self.dilation[i], stride=self.stride, padding=self.dilation[i]),
            nn.BatchNorm2d(self.f),
            nn.ReLU(),
            nn.Conv2d(self.f, self.f, kernel_size=self.kernel, dilation=self.dilation[i], stride=self.stride, padding=self.dilation[i])
        ]))
        

        if isinstance(normal_sides, (list)):
            self.normal_side = nn.Sequential(*nn.ModuleList(normal_sides)) # MUST USE MODULE LIST OR WEIGHTS WON'T MOVE TO DEVICE
        else:
            self.normal_side = [nn.Sequential(*nn.ModuleList([normal_sides[i] for i in range(len(normal_sides))]))]
        
        
        skip_sides = [
            nn.Conv2d(in_channels=self.f,
                      out_channels=self.f,
                      kernel_size=self.kernel,
                      dilation=self.dilation[0],
                      stride=self.stride,
                      padding=1),
            nn.BatchNorm2d(self.f)
        ]
        
        for i in range(1, len(dilation)):
            skip_sides = np.vstack((skip_sides, [
            nn.Conv2d(in_channels=self.f,
                      out_channels=self.f,
                      kernel_size=self.kernel,
                      dilation=self.dilation[i],
                      stride=self.stride,
                      padding=self.dilation[i]),
            nn.BatchNorm2d(self.f)
        ]))
        
        if isinstance(skip_sides, (list)):
            self.skip_side = nn.Sequential(*nn.ModuleList(skip_sides))
        else:
            self.skip_side = [nn.Sequential(*nn.ModuleList([skip_sides[i] for i in range(len(skip_sides))]))]
    
    def forward(self, x):
        #normal_out = torch.cat(*[self.normal_side[i](x) for i in range(len(self.dilation))], dim=0)
        #skip_out = torch.cat([self.skip_side[i](x) for i in range(len(self.dilation))], dim=0)
        
        normal_out = self.normal_side(x)
        skip_out = self.skip_side(x)
        add = normal_out + skip_out
        
        return add
    
    
if __name__ == '__main__':
    t = torch.randn([2,3,256,256], dtype=torch.float)
    pool = ResUNet()
    
    out = pool(t)
    print(out.shape)
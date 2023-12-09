# Code by Nicole, but not my model https://github.com/feevos/resuneta/tree/master

import torch
import torch.nn as nn
import numpy as np

# mx code adapted to pytorch from https://github.com/feevos/resuneta/tree/master
class ResUNet(nn.Module):
    '''
    ResUNet model
    
    INPUTS:
    num_classes: Number of classes to identify
    depth: Number of steps to encode and decode
    nFilters: Starting amount of filters
    '''
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
                      stride=1),
            nn.BatchNorm2d(self.nFilters) # actual code uses a batch norm for its first conv layer, but paper doesn't reference it
            # https://github.com/feevos/resuneta/blob/master/models/resunet_d6_encoder.py#L36
        ])
        
        # The dilations in the code did not match what was in the paper. They use 1, 3, 15 often instead of 1, 3, 15, 31 schemes like in the paper
        self.layer_2 = nn.Sequential(*[
            ResBlock(features=self.nFilters, kernel=3, dilation=[1, 3, 15], stride=1)
        ])
        
        self.layer_3_4 = nn.Sequential(*[
            nn.Conv2d(in_channels=self.nFilters * 2 ** 0,
                      out_channels=self.nFilters * 2 ** 1,
                      kernel_size=1,
                      dilation=1,
                      stride=2),
            ResBlock(features=self.nFilters * 2 ** 1, kernel=3, dilation=[1, 3, 15], stride=1)
        ])
        
        self.layer_5_6 = nn.Sequential(*[
            nn.Conv2d(in_channels=self.nFilters * 2 ** 1,
                      out_channels=self.nFilters * 2 ** 2,
                      kernel_size=1,
                      dilation=1,
                      stride=2),
            ResBlock(features=self.nFilters * 2 ** 2, kernel=3, dilation=[1, 3, 15], stride=1)
        ])
        
        
        self.layer_7_8 = nn.Sequential(*[
            nn.Conv2d(in_channels=self.nFilters * 2 ** 2,
                      out_channels=self.nFilters * 2 ** 3,
                      kernel_size=1,
                      dilation=1,
                      stride=2),
            ResBlock(features=self.nFilters * 2 ** 3, kernel=3, dilation=[1, 3], stride=1)
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
        
        self.psp_13 = PSP_Pooling(self.nFilters * 2 ** 5)
        
        self.layer_14 = nn.Sequential(*[
            nn.Upsample(scale_factor=2),
            UpConv(512)
        ])
        
        self.layer_15_16 = nn.Sequential(*[
            Combine(512),
            ResBlock(512)
        ])
        
        self.layer_17 = nn.Sequential(*[
            nn.Upsample(scale_factor=2),
            UpConv(256)
        ])
        
        self.layer_18_19 = nn.Sequential(*[
            Combine(256),
            ResBlock(256)
        ])
        
        self.layer_20 = nn.Sequential(*[
            nn.Upsample(scale_factor=2),
            UpConv(128)
        ])
        
        self.layer_21_22 = nn.Sequential(*[
            Combine(128),
            ResBlock(128)
        ])
        
        self.layer_23 = nn.Sequential(*[
            nn.Upsample(scale_factor=2),
            UpConv(64)
        ])
        
        self.layer_24_25 = nn.Sequential(*[
            Combine(64),
            ResBlock(64)
        ])
        
        self.layer_26 = nn.Sequential(*[
            nn.Upsample(scale_factor=2),
            UpConv(32)
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
            PSP_Pooling(32),
            nn.Conv2d(in_channels=32, out_channels=self.num_classes, kernel_size=1, dilation=1, stride=1)
        ])
        
        self.out = nn.Softmax(dim=1)
        '''
        Original layout from paper (does not perfectly match the implementation on the GitHub)
        
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
    '''
    Combines two tensors
    
    INPUTS:
    features: Number of features in one tensor
    '''
    def __init__(self, features):
        super().__init__()
        self.features = features
        
        self.relu = nn.ReLU()
        self.conv2d = nn.Conv2d(in_channels=self.features * 2, out_channels=self.features, kernel_size=1, dilation=1, stride=1)
        self.norm = nn.BatchNorm2d(self.features)

    def forward(self, x):
        input1 = x[0]
        input2 = x[1]
        
        input1 = self.relu(input1)
        input2 = torch.cat((input1, input2), dim=1)
        
        return self.norm(self.conv2d(input2)) # batch norm conv, equivalent to Conv2dNormed in original code

class UpConv(nn.Module):
    '''
    Convolution for upsampled images
    
    INPUTS:
    features: Number of output features
    '''
    def __init__(self, features):
        super().__init__()
        self.output_features = features
        self.input_features = features * 2
        
        self.conv2d = nn.Conv2d(in_channels=self.input_features, out_channels=self.output_features,
                                kernel_size=1)
        
    def forward(self, x):
        return self.conv2d(x)
    
class PSP_Pooling(nn.Module):
    '''
    MOSTLY NOT MY CODE ASIDE FROM VERY MINOR CHANGES
    Translated algorithm from mxnet to pytorch: https://github.com/feevos/resuneta/blob/master/nn/pooling/psp_pooling.py
    '''
    def __init__(self, nfilters, depth = 4):
        super().__init__()
                
        
        self.nfilters = nfilters
        self.depth = depth 
        
        # This is used as a container (list) of layers
        self.convs = nn.Sequential()
        for _ in range(depth):
            self.convs.add_module(str(_), nn.Sequential(
                nn.Conv2d(self.nfilters, self.nfilters//self.depth,kernel_size=1,padding=0),
                nn.BatchNorm2d(self.nfilters//self.depth),
                nn.ReLU()
                ))
            
        self.conv_norm_final = nn.Sequential(
                        nn.Conv2d(self.nfilters * 2, self.nfilters, 1),
                        nn.BatchNorm2d(self.nfilters),
                        nn.ReLU()
                        )
        
        self.pool = nn.AdaptiveMaxPool2d(1)


    # ******** Utilities functions to avoid calling infer_shape ****************
    def HalfSplit(self,_a):
        """
        Returns a list of half split arrays. Usefull for HalfPoolling 
        """
        b  = torch.split(_a,_a.shape[2] // 2, dim=2) # Split First dimension 
        c1 = torch.split(b[0], b[0].shape[3] // 2, dim=3) # Split 2nd dimension
        c2 = torch.split(b[1], b[1].shape[3] // 2, dim = 3) # Split 2nd dimension
    
    
        d11 = c1[0]
        d12 = c1[1]
    
        d21 = c2[0]
        d22 = c2[1]
    
        return [d11,d12,d21,d22]
    
    
    def QuarterStitch(self,_Dss):
        """
        INPUT:
            A list of [d11,d12,d21,d22] block matrices.
        OUTPUT:
            A single matrix joined of these submatrices
        """
    
        temp1 = torch.cat((_Dss[0],_Dss[1]),dim=-1)
        temp2 = torch.cat((_Dss[2],_Dss[3]),dim=-1)
        result = torch.cat((temp1,temp2),dim=2)

        return result
    
    
    def HalfPooling(self,_a):
        """
        Tested, produces consinstent results.
        """
        Ds = self.HalfSplit(_a)
    
        Dss = []
        for x in Ds:
            Dss += [torch.mul(torch.ones_like(x) , self.pool(x))]
     
        return self.QuarterStitch(Dss)    
      
    

    #from functools import lru_cache
    #@lru_cache(maxsize=None) # This increases by a LOT the performance 
    # Can't make it to work with symbol though (yet)
    def SplitPooling(self, _a, depth):
        #print("Calculating F", "(", depth, ")\n")
        """
        A recursive function that produces the Pooling you want - in particular depth (powers of 2)
        """
        if depth==1:
            return self.HalfPooling(_a)
        else :
            D = self.HalfSplit(_a)
            return self.QuarterStitch([self.SplitPooling(d,depth-1) for d in D])

        
    # ***********************************************************************************  

    def forward(self,_input):

        p  = [_input]
        # 1st:: Global Max Pooling . 
        p += [self.convs[0](torch.mul(torch.ones_like(_input) , self.pool(_input)))]
        p += [self.convs[d](self.SplitPooling(_input,d)) for d in range(1,self.depth)]
        out = torch.cat(p,dim=1)
        out = self.conv_norm_final(out)

        return out
    
'''
Here lies some previous disasterous attempts to figure out how to write PSP pooling myself.
I ultimately just translated their code from mxnet to pytorch

# ResUNet uses Pyramid Pooling https://arxiv.org/pdf/1612.01105.pdf]
class PyramidPooling(nn.Module):
    def __init__(self, features, out_channels, size):
        super(PyramidPooling, self).__init__()
        self.f = features

        self.conv1 = nn.Sequential(nn.Conv2d(self.f, self.f, kernel_size=1),
                                    nn.BatchNorm2d(self.f),
                                    nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(self.f, self.f // 2, kernel_size=1),
                                    nn.BatchNorm2d(self.f // 2),
                                    nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(self.f, self.f // 4, kernel_size=1),
                                    nn.BatchNorm2d(self.f // 4),
                                    nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(self.f, self.f // 8, kernel_size=1),
                                    nn.BatchNorm2d(self.f // 8),
                                    nn.ReLU())
        
        self.out = nn.Sequential(nn.Conv2d(self.f * 2, out_channels, kernel_size=1),
                                    nn.BatchNorm2d(out_channels),
                                    nn.ReLU())
        
        # The paper claimed to have split along the fe
        self.pool = nn.MaxPool2d(size)
    def forward(self, x):
        quarter_split = x.split()
        # pool and reconstruct shape
        feat1 = torch.cat([torch.cat([self.pool(quarter_split[i][j]) for j in range(len(quarter_split))], dim=3) for i in range(len(half_split))], dim = 2)
        feat2 = torch.cat([torch.cat([self.pool(quarter_split[i][j]) for j in range(len(quarter_split))], dim=3) for i in range(len(half_split))], dim = 2)
        feat3 = torch.cat([torch.cat([self.pool(quarter_split[i][j]) for j in range(len(quarter_split))], dim=3) for i in range(len(half_split))], dim = 2)
        feat4 = torch.cat([torch.cat([self.pool(quarter_split[i][j]) for j in range(len(quarter_split))], dim=3) for i in range(len(half_split))], dim = 2)
        
        
        
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
                
        return x
    
    def split(self, x):
        # split images
        half_split = torch.split(x, int(x.shape[2] / 2), dim=2)
        quarter_split = [torch.split(half_split[i], int(x.shape[3] / 2), dim=3) for i in range(len(half_split))]
        return quarter_split
    
class PyramidPooling(nn.Module):
    Alternate Pyramid Pooling for d7 middle pooling layer
    PSEUDOCODE
    CONV2D features, k=1,d=1,s=2
    ResBlock features, k=3
    MaxPool kernel=2 stride=2
    Upsample
    Concat Upsample and ResBlock
    Conv2D features k=1
    
    I noticed that the code here does not have a stride of 2 as referenced by the paper, but upsamples the low layer:
    https://github.com/feevos/resuneta/blob/49d26563f84c737e07d34edfe30b56c59cbb4203/nn/pooling/psp_pooling.py#L22

    It uses a stride of 1 with the normed conv2d:
    https://github.com/feevos/resuneta/blob/49d26563f84c737e07d34edfe30b56c59cbb4203/nn/layers/conv2Dnormed.py#L13

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
            #https://github.com/feevos/resuneta/blob/49d26563f84c737e07d34edfe30b56c59cbb4203/nn/layers/scale.py#L66
            # They want to use Bilinear, but the actual implementation uses nearest bc programming problems. I'll do bilinear
            nn.Upsample(scale_factor=2)
        ])
        
        self.conv_out = nn.Conv2d(in_channels=self.nFeatures * 2,
                                  out_channels=self.nFeatures,
                                  kernel_size=1,
                                  dilation=1,
                                  stride=1)
        
        # Paper doesn't use Conv2dNormed, but code does:
        # https://github.com/feevos/resuneta/blob/49d26563f84c737e07d34edfe30b56c59cbb4203/nn/pooling/psp_pooling.py#L24
        self.norm = nn.BatchNorm2d(self.nFeatures)
        
    def forward(self, x):
        layer_b = self.layer_B(x)
        layer_d = self.layer_D(layer_b)
        concat = torch.cat((layer_d, layer_b), dim=1)
        out = self.norm(self.conv_out(concat)) # batch norm conv, equivalent to Conv2dNormed in original code
        return out
'''

        
class ResBlock(nn.Module):
    '''
    Residual connection convolution
    
    INPUTS:
    features: number of input features
    kernel: size of kernel
    dilation: list of dilation sizes
    stride: size of stride
    '''
    def __init__(self, features, kernel = 3, dilation=[1, 3, 15], stride=1):
        super().__init__()
        self.f = features
        self.kernel = kernel
        self.dilation = dilation
        self.stride = stride
        

        # NOTE: It's coincidental the dilation is the same as the padding
        # Our kernel size is always 3, which lets the shape math work out...
        normal_sides = [
            nn.BatchNorm2d(self.f),
            nn.ReLU(),
            nn.Conv2d(self.f, self.f,
                      kernel_size=self.kernel,
                      dilation=self.dilation[0],
                      stride=self.stride,
                      padding=self.dilation[0]),
            nn.BatchNorm2d(self.f),
            nn.ReLU(),
            nn.Conv2d(self.f, self.f,
                      kernel_size=self.kernel,
                      dilation=self.dilation[0],
                      stride=self.stride,
                      padding=self.dilation[0])
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
            self.normal_side = nn.ModuleList([nn.Sequential(*nn.ModuleList(normal_sides))]) # MUST USE MODULE LIST OR WEIGHTS WON'T MOVE TO DEVICE
        else:
            self.normal_side = nn.ModuleList([nn.Sequential(*nn.ModuleList(normal_sides[i])) for i in range(len(normal_sides))])
        
        '''
        # Image reference for ResBlock skip: https://www.researchgate.net/figure/Flowchart-of-the-resblock-Each-resblock-is-composed-of-a-batch-normalization-a_fig2_330460151

        skip_sides = [
            nn.Conv2d(in_channels=self.f,
                      out_channels=self.f,
                      kernel_size=self.kernel,
                      dilation=self.dilation[0],
                      stride=self.stride,
                      padding=self.dilation[0]),
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
            self.skip_side = nn.ModuleList([nn.Sequential(*nn.ModuleList(skip_sides))]) # MUST USE MODULE LIST OR WEIGHTS WON'T MOVE TO DEVICE
        else:
            self.skip_side = nn.ModuleList([nn.Sequential(*nn.ModuleList(skip_sides[i])) for i in range(len(skip_sides))])
        '''
    def forward(self, x):
        # Make all dilations sizes sides and add them together
        normal_out = self.normal_side[0](x)
        
        # ResUNet uses modified residual blocks that do not have a skip side... which I realized AFTER writing this
        #skip_out = self.skip_side[0](x)
        
        for i in range(1, len(self.dilation)):
            normal_out += self.normal_side[i](x)
            #skip_out += self.skip_side[i](x)

        #add = normal_out + skip_out
        add = normal_out + x # modified residuals blocks uses the input x as the residual connection w/ no extra layers
        return add
    
    
if __name__ == '__main__':
    device = 'cpu'
    t = torch.randn([2,1024,8,8], dtype=torch.float)
    pool = PSP_Pooling(1024).to(device)
    
    out = pool(t)
    print(out.shape)
    
    t = torch.randn([2,3,256,256], dtype=torch.float)
    resunet = ResUNet()
    out = resunet(t)
    print(out.shape)
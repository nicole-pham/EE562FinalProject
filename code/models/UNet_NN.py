# Code by Sean

# From Nicole: Sean pushed this. He said he wrote it, but he didn't mention his source
# Based on a web search, I believe he used code from here: https://towardsdatascience.com/cook-your-first-u-net-in-pytorch-b3297a844cf3

import torch
import torch.nn as nn
import torch.nn.functional as F

N_CLASS = 6

# NOTE: output_width = (input_width - kernel_size + 2* padding)/stride + 1
class UNet(nn.Module):
    def __init__(self):
      super(UNet, self).__init__()
      # Encoder (Down Convolution)
      # input: 256x256x3
      self.conv00 = nn.Conv2d(3,64,kernel_size=3,padding=1)  # out: 254x254x64
      self.conv01 = nn.Conv2d(64,64,kernel_size=3,padding=1) # out: 252x252x64
      self.pool0 = nn.MaxPool2d(kernel_size=2, stride=2)     # out: 126x126x64

      # input: 126x126x64
      self.conv10 = nn.Conv2d(64,128,kernel_size=3,padding=1)  #out:124x124x128
      self.conv11 = nn.Conv2d(128,128,kernel_size=3,padding=1) #out:122x122x128
      self.conv12 = nn.Conv2d(128,128,kernel_size=3,padding=1)  #120x120x128    
      self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)       #out:60x60x128

      # input: 60x60x128
      self.conv20 = nn.Conv2d(128,256,kernel_size=3,padding=1)   #out:58x58x256
      self.conv21 = nn.Conv2d(256,256, kernel_size=3, padding=1) #out:56x56x256
      self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)         #out:28x28x256
      
      #input: 28x28x256
      self.conv30 = nn.Conv2d(256,512, kernel_size=3, padding=1) #out:26x26x512
      self.conv31 = nn.Conv2d(512,512, kernel_size=3, padding=1) #out:24x24x512
      self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)         #out:12x12x512

      # input: 12x12x512
      self.conv40 = nn.Conv2d(512,1024,kernel_size=3,padding=1)  #out:10x10x1024
      self.conv41 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)#out:8x8x1024

      # Decoder (Up convolution)
      # input 8x8x1024
      self.upconv0 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
      self.up40 = nn.Conv2d(1024, 512, kernel_size=3, padding=1) 
      self.up41 = nn.Conv2d(512, 512, kernel_size=3, padding=1) 

      self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
      self.up30 = nn.Conv2d(512, 256, kernel_size=3, padding=1) 
      self.up31 = nn.Conv2d(256, 256, kernel_size=3, padding=1) 

      self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
      self.up20 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
      self.up21 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

      self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
      self.up10 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
      self.up11 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

      # Output layer
      self.outconv = nn.Conv2d(64, N_CLASS, kernel_size=1)

    def forward(self, x):
         # Encoder (Down Conv)
         # input 256x256x3
         xe11 = F.relu(self.conv00(x))    #out: 254x254x64
         xe12 = F.relu(self.conv01(xe11)) #out: 252x252x64
         xp1 = self.pool0(xe12)           #out: 126x126x64

         xe21 = F.relu(self.conv10(xp1))  #out: 124x124x128
         xe22 = F.relu(self.conv11(xe21)) #out: 122x122x128
         xe23 = F.relu(self.conv12(xe22)) #out: 120x120x128
         xp2 = self.pool1(xe23)           #out: 60x60x128

         xe31 = F.relu(self.conv20(xp2))  #out: 58x58x256
         xe32 = F.relu(self.conv21(xe31)) #out: 56x56x256
         xp3 = self.pool2(xe32)           #out: 28x28x256

         xe41 = F.relu(self.conv30(xp3))  #out: 26x26x512
         xe42 = F.relu(self.conv31(xe41)) #out: 24x24x512
         xp4 = self.pool3(xe42)           #out: 12x12x512

         xe51 = F.relu(self.conv40(xp4))  #out: 10x10x1024
         xe52 = F.relu(self.conv41(xe51)) #out: 8x8x1024
         
         # Decoder (Up Conv)
         xu1 = self.upconv0(xe52)         
         xu11 = torch.cat([xu1, xe42], dim=1) #out: 16x16x1024 
         xd11 = F.relu(self.up40(xu11))       #out: 14x14x512
         xd12 = F.relu(self.up41(xd11))       #out: 12x12x512

         xu2 = self.upconv1(xd12)         
         xu22 = torch.cat([xu2, xe32], dim=1) #out: 24x24x512
         xd21 = F.relu(self.up30(xu22))       #out: 22x22x256
         xd22 = F.relu(self.up31(xd21))       #out: 20x20x256

         xu3 = self.upconv2(xd22)             
         xu33 = torch.cat([xu3, xe23], dim=1) #out: 40x40x256
         xd31 = F.relu(self.up20(xu33))       #out: 38x38x128
         xd32 = F.relu(self.up21(xd31))       #out: 36x36x128

         xu4 = self.upconv3(xd32)             
         xu44 = torch.cat([xu4, xe12], dim=1) #out: 72x72x128
         xd41 = F.relu(self.up10(xu44))       #out: 70x70x64
         xd42 = F.relu(self.up11(xd41))       #out: 68x68x64

         # Output layer
         out = self.outconv(xd42)             #out: 68 x 68 x 6 classes

         return out


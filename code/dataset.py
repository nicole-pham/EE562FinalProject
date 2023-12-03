import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import os
import glob
from torchvision.transforms import ToPILImage
import torch.nn.functional as F
import numpy as np

import random

# Code heavily inspired by https://github.com/Eladamar/fast_scnn/blob/master/utils/dataset.py
# as well as https://github.com/zyxu1996/Efficient-Transformer/blob/main/dataset.py
ClassesColors_ColorKey = {
    (255,0,0):0, # Red: Background
    (255,255,255):1, # White: Impervious surfaces
    (255,255,0):2, # Yellow: Cars
    (0,0,255):3, # Blue: Buildings
    (0,255,255):4, # Cyan: Low Vegetation
    (0,255,0):5 # Green: Trees
}

ClassesColors_IndexKey = {
    0: (255,0,0), # Red: Background
    1:(255,255,255), # White: Impervious surfaces
    2:(255,255,0), # Yellow: Cars
    3:(0,0,255), # Blue: Buildings
    4:(0,255,255), # Cyan: Low Vegetation
    5:(0,255,0) # Green: Trees
}

class PotsdamDataset(Dataset):
    def __init__(self, image_dir, transform = True):
        self.image_paths = glob.glob(os.path.join(image_dir, '*npy')) # get image paths (stored as numpy arrays)
                
    def __len__(self):
        return len(self.image_paths) # length is how many images we have in the dataset
    
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        img = np.load(image_path)
        
        # Load mask
        mask_path = image_path.replace('.npy', '-mask.npy')
        mask_path = mask_path.replace('imgs', 'masks')
        mask = np.load(mask_path)

        if self.transform:
            img, mask = self.transform(img, mask)
        else:
            img = torch.tensor(img, dtype=torch.float64)
            mask = torch.tensor(mask, dtype=torch.float64)
        
        
        return img, mask # img shape: colorsxheightxwidth, mask shape: class_nxheightxwidth
    
    def transform(self, img, mask):
        #mask = self.ToRGB(mask)
        
        
        affine_transform = transforms.Compose([
            RandomHorizontalFlip(0.5)
        ])
        
        normal_transform = transforms.Compose([
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
        
        
        transformed_img, transformed_mask = affine_transform((img, mask))
        transformed_img = torch.tensor(transformed_img.copy() / 255.0)
        transformed_img = normal_transform(transformed_img)
        
        #transformed_mask = self.ToMask(transformed_mask)
        return transformed_img, torch.tensor(transformed_mask.copy(), dtype=torch.float64)
    
    def ToRGB(self, mask):
        indicies = np.argmax(mask, axis=0) # get which class each pixel is
        
        mask_RGB = [[ClassesColors_IndexKey[indicies[x][y]]
                    for x in range(indicies.shape[0])]
                    for y in range(indicies.shape[1])]
        
        return np.array(mask_RGB) # shape: colorsxheightxwidth
    
    def ToMask(self, RGB):
        mask = [[ClassesColors_ColorKey[tuple(RGB[x,y,:])]
                    for x in range(RGB.shape[0])]
                    for y in range(RGB.shape[1])]
        
        return mask
    
class RandomHorizontalFlip:
    def __init__(self, prob):
        self.prob = prob
        
    def __call__(self, imgmask):
        img, mask = imgmask
        rand = random.random()
        if rand < self.prob:
            img = np.flip(img, axis=2)
            mask = np.flip(mask, axis=2)
        
        return img, mask
        

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    
    dataset = PotsdamDataset('../data/Potsdam_6k/training/imgs')
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    for i, sample in enumerate(dataloader):
        img = sample[0]
        mask = sample[1]
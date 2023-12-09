# Code by Nicole

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import os
import glob
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
    '''
    Dataset object for loading Potdam images in .npy reprensentations. This dataset assumes
    your imgs are stored in a folder called imgs and your masks in a folder called masks
    in a structure similar to this:

    |home_dir
        |data_dir
            | imgs
            | masks

    INPUTS:
    image_dir: The location your IMAGES are stored (e.g. home_dir/data_dir/imgs)
    transform: (True by default) Whether to or not to transform the images

    OUTPUTS:
    img: The loaded and possibly transformed image
    mask: The associated label to go with the img
    '''
    def __init__(self, image_dir, transform = True):
        self.image_paths = glob.glob(os.path.join(image_dir, '*npy')) # get image paths (stored as numpy arrays)
        self.flip = transform
        
    def __len__(self):
        return len(self.image_paths) # length is how many images we have in the dataset
    
    def __getitem__(self, idx):
        # ONLY LOAD IMAGES DURING GET ITEM, NOT INIT! UNLESS YOUR COMPUTER IS BEEFY ENOUGH TO LOAD ALL THOSE IMAGES AT ONCE

        # Load image
        image_path = self.image_paths[idx]
        img = np.load(image_path)
        
        # Load mask
        mask_path = image_path.replace('.npy', '-mask.npy')
        mask_path = mask_path.replace('imgs', 'masks')
        mask = np.load(mask_path)

        img, mask = self.transform(img, mask, self.flip)

        return img, mask # img shape: colorsxheightxwidth, mask shape: class_nxheightxwidth
    
    def transform(self, img, mask, flip=True):
        '''
        Modifies loaded images to being usable for training

        INPUTS:
        img: colorChannelsxHeightxWidth ndarray of the image
        mask: nClassesxHeightxWidth ndarray of the labels

        OUTPUTS:
        transformed_img: colorChannelsxHeightxWidth ndarray of the transformed image
        transformed_mask: nClassesxHeightxWidth ndarray of the labels with matching img transform
        '''
        
        # Randomly flips images horizontally and/or vertically so the training
        # doesn't see the exact same images in successive epochs
        affine_transform = transforms.Compose([
            RandomHorizontalFlip(0.5),
            RandomVerticalFlip(0.5)
        ])
        
        # Takes image from 0 to 1 to -1 to 1
        normal_transform = transforms.Compose([
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
        ])
        
        if flip:
            # Flip the images and masks together
            transformed_img, transformed_mask = affine_transform((img, mask))
        else:
            transformed_mask = mask

        # Essentially doing torch.ToTensor without changing dims since they're already in CxHxW
        transformed_img = torch.tensor(transformed_img.copy() / 255.0)

        # transformed_img = torch.tensor(transformed_img.copy(), dtype=torch.float64)
        transformed_img = normal_transform(transformed_img)
        
        return transformed_img, torch.tensor(transformed_mask.copy(), dtype=torch.float64)
    
    def ToRGB(self, mask):
        '''
        Helper function for turning class masks into their RGB representations

        INPUTS:
        mask: nClassesxHeightxWidth ndarray holding class mapping

        OUTPUTS:
        mask_RGB: colorChannelsxHeightxWidth ndarray for RGB represenation of mask
        '''
        indicies = np.argmax(mask, axis=0) # get which class each pixel is
        
        mask_RGB = [[ClassesColors_IndexKey[indicies[x][y]]
                    for x in range(indicies.shape[0])]
                    for y in range(indicies.shape[1])]
        
        return np.array(mask_RGB) # shape: colorsxheightxwidth
    
    def ToMask(self, RGB):
        '''
        Helper function for turning RGB reprentations of class masks into masks for training

        INPUTS:
        RGB: colorChannelsxHeightxWidth ndarray for RGB represenation of mask

        OUTPUTS:
        mask: nClassesxHeightxWidth ndarray holding class mapping
        '''
        mask = [[ClassesColors_ColorKey[tuple(RGB[x,y,:])]
                    for x in range(RGB.shape[0])]
                    for y in range(RGB.shape[1])]
        
        return mask
    
class RandomHorizontalFlip:
    '''
    Custom horizontal flip transform to flip the image and mask at the same time

    INIT:
    prob: float from [0, 1] for the probability it will flip

    CALL:
    imgmask: list of [img, mask] (both img and mask are ndarrays)

    OUTPUTS:
    img: randomly flipped img ndarray
    mask: randomly flipped mask ndarray
    '''
    def __init__(self, prob):
        self.prob = prob
        
    def __call__(self, imgmask):
        img, mask = imgmask
        rand = random.random()
        if rand < self.prob:
            img = np.flip(img, axis=2)
            mask = np.flip(mask, axis=2)
        
        return img, mask
        
class RandomVerticalFlip:
    '''
    Custom vertical flip transform to flip the image and mask at the same time

    INIT:
    prob: float from [0, 1] for the probability it will flip

    CALL:
    imgmask: list of [img, mask] (both img and mask are ndarrays)

    OUTPUTS:
    img: randomly flipped img ndarray
    mask: randomly flipped mask ndarray
    '''
    def __init__(self, prob):
        self.prob = prob
    
    def __call__(self, imgmask):
        img, mask = imgmask
        rand = random.random()
        if rand < self.prob:
            img = np.flip(img, axis=1)
            mask = np.flip(mask, axis=1)

        return img, mask



if __name__ == '__main__':
    # For unit testing
    from torch.utils.data import DataLoader
    
    dataset = PotsdamDataset('../../data/Potsdam_6k/training/imgs')
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    for i, sample in enumerate(dataloader):
        img = sample[0]
        mask = sample[1]
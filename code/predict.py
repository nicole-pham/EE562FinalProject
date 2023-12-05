import os
import argparse
import torch

from models.FastSCNN import FastSCNN
from models.swinT import swin_tiny as swinT
from models.resunet import ResUNet

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from dataset import RandomHorizontalFlip

from torchvision import transforms

# Taken from https://github.com/Eladamar/fast_scnn/blob/master/metrics.py
def pixel_accuracy(preds, mask):
    preds = preds[0]
    pred_argmax = torch.argmax(preds, axis=1)
    mask_argmax = torch.argmax(mask, axis=1)
    acc_sum = torch.sum(pred_argmax == mask_argmax).item()
    # divide by batch size
    acc_mean = acc_sum / preds.shape[0]
    image_size = preds.shape[2] * preds.shape[3]
    acc = float(acc_mean) / (image_size + 1e-10)
    return acc

parser = argparse.ArgumentParser(
    description='Predict segmentation result from a given image')
parser.add_argument('--checkpoint', type=str, default='checkpoints/SwinT12_03',
                    help='which checkpoint to use')
parser.add_argument('--input_image', type=str,
                    help='path to the input picture')
parser.add_argument('--outdir', default='data/test_result', type=str,
                    help='path to save the predict result')
parser.add_argument('--num_classes', type=int, default=6,
                    help='num of classes in model')
parser.add_argument('--eval', default=None, type=str,
                    help='image label for evaluation score')


args = parser.parse_args()

def predict():
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    # output folder
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    #img = np.load(args.input_image)
    img = np.load('img-0a0d7c9d-1158-4d74-8c6b-6f536ebca85f.npy')
    mask = np.load('img-0a0d7c9d-1158-4d74-8c6b-6f536ebca85f-mask.npy')
    
    affine_transform = transforms.Compose([
        RandomHorizontalFlip(1)
    ])
    
    normal_transform = transforms.Compose([
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
        
        
    transformed_img, transformed_mask = affine_transform((img, mask))
    transformed_img = torch.tensor(transformed_img.copy() / 255.0)
    normal_transformed_img = normal_transform(transformed_img)
    transformed_mask = torch.tensor(transformed_mask.copy(), dtype=torch.float64)
    
    fig = plt.figure(1)
    canvas = FigureCanvas(fig)
    plt.imshow(torch.transpose(torch.transpose(transformed_img, 0, 2), 0, 1))
    canvas.print_figure('test-pic.png')
    
    #model = FastSCNN(args.num_classes).to(device)
    #model = swinT(nclass=args.num_classes, pretrained=True, aux=True, head="mlphead", edge_aux=False)
    model = ResUNet()

    model.load_state_dict(torch.load(args.checkpoint, map_location=device), strict=False)
    # model = torch.load(args.checkpoint, map_location=device)
    print('Finished loading model!')
    model.eval()
    model.double()
    with torch.no_grad():
        outputs = model(normal_transformed_img.unsqueeze(0).to(device))
    pred = torch.argmax(outputs[0], 0).to(device)
    
    transformed_mask = np.argmax(transformed_mask, 0)
    
    ClassesColors = {
        (255,0,0):0,
        (255,255,255):1,
        (255,255,0):2,
        (0,0,255):3,
        (0,255,255):4,
        (0,255,0):5
    }
    
    Class1H2RGB =  dict([[str(val),key] for key,val in ClassesColors.items()])
    cmap = ListedColormap(np.array([Class1H2RGB[k] for k in sorted(Class1H2RGB.keys())])/255.0)
    fig = plt.figure(2)
    canvas = FigureCanvas(fig)
    plt.imshow(transformed_mask,cmap=cmap,rasterized=True)
    canvas.print_figure('test-mask.png')
    fig = plt.figure(3)
    canvas = FigureCanvas(fig)
    plt.imshow(pred, cmap=cmap, rasterized=True)
    canvas.print_figure('test-pred.png')
    plt.show(block=True)

    if not args.eval is None:
        # only one label for now, so add batch=1 dim
        label = torch.load(args.eval).unsqueeze(0)
        output = outputs[0]
        print("image score: ", pixel_accuracy(output, label))


if __name__ == '__main__':
    predict()


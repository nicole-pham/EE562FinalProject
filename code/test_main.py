# Code originally by Justin, revisions by Nicole (changed scores from taking the 
# average of the last batch to average over the whole validation set, heavy refactoring)
# See GitHub commit for exact changes.

import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix

from dataset import PotsdamDataset
from models.swinT import swin_tiny as swinT
from models.resunet import ResUNet
from models.UNet_NN import UNet
from models.FastSCNN import FastSCNN

# Function to compute F1 score
def f1_score_metric(preds, mask):
    preds = torch.argmax(preds, dim=1).view(-1).cpu()
    mask = torch.argmax(mask, dim=1).view(-1).cpu()
    f1 = f1_score(mask, preds, average='macro')
    return f1

# Function to compute mIoU
def compute_iou(pred, mask):
    pred = torch.argmax(pred, dim=1).cpu().numpy().flatten()
    mask = torch.argmax(mask, dim=1).cpu().numpy().flatten()
    cm = confusion_matrix(mask, pred)
    intersection = np.diag(cm)
    union = np.sum(cm, axis=0) + np.sum(cm, axis=1) - np.diag(cm)
    iou = np.mean(intersection / (union + 1e-10))
    return iou

# Taken from https://github.com/Eladamar/fast_scnn/blob/master/metrics.py
def pixel_accuracy(preds, mask):
    preds = preds
    pred_argmax = torch.argmax(preds, axis=1)
    mask_argmax = torch.argmax(mask, axis=1)
    acc_sum = torch.sum(pred_argmax == mask_argmax).item()
    # divide by batch size
    acc_mean = acc_sum / preds.shape[0]
    image_size = preds.shape[2] * preds.shape[3]
    acc = float(acc_mean) / (image_size + 1e-10)
    return acc

def test(dataloader, model, loss_fn, acc):
    '''
    Function to get metrics over validation set
    
    INPUTS:
    dataloader: dataloader with your validation set
    model: model to test
    loss_fn: loss function (usually cross entropy loss)
    acc: function for finding pixel accuracy
    '''
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, f1, pixel_accuracy, iou = 0, 0, 0, 0
    count = 0
    with torch.no_grad():
        for X, y in dataloader:
            count += 1
            X, y = X.to(torch.float32), y.to(torch.float32)
            X, y = X.to(device), y.to(device)
            pred = model(X)
            
            # sum all the scores
            if isinstance(pred, (list, tuple)):
                for i in range(len(pred)):
                    test_loss += loss_fn(pred[i], y).item()
                pixel_accuracy += acc(pred[0], y)
                f1 += f1_score_metric(pred[0], y)
                iou += compute_iou(pred[0], y)
            else:
                test_loss += loss_fn(pred, y).item()
                pixel_accuracy += acc(pred, y)
                f1 += f1_score_metric(pred, y)
                iou += compute_iou(pred, y)
                
    # take average of scores
    test_loss /= num_batches
    pixel_accuracy /= num_batches
    f1 /= num_batches
    iou /= num_batches

    print(f"loss: {test_loss:>7f}  accuracy: {pixel_accuracy:>7f}  F1: {f1:>7f}  mIoU: {iou:>7f}")

# code heavily borrowed from https://github.com/Eladamar/fast_scnn/blob/master/main.py
batch_size = 16 # increased batch size (images are 256x256 instead of 6000x6000)
device = 'cuda'

test_image_path = '../../data/Potsdam_6k/validation/imgs'

ds_test = PotsdamDataset(test_image_path, transform=False)

dl_test = DataLoader(ds_test, batch_size, shuffle=False)

# Code reference: https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
#model = ResUNet()
#model = swinT(nclass=6, pretrained=True, aux=True, head="mlphead", edge_aux=False)
#model = UNet()
model = FastSCNN(num_classes=6)

model_state_dict = torch.load('./checkpoints/FastSCNN10_epoch')
model.load_state_dict(model_state_dict)

loss_fn = CrossEntropyLoss()
success_metric = pixel_accuracy

model.to(device)
test(dl_test, model, loss_fn, success_metric)
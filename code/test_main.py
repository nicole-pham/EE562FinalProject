# Code by Justin

import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import torchvision.transforms.functional as F
import numpy as np
import datetime
from sklearn.metrics import f1_score, confusion_matrix

from dataset import PotsdamDataset
from FastSCNN import FastSCNN
from models.swinT import swin_tiny as swinT
from models.customCNN import CustomCNN
from models.unet import UNet


# Function to compute F1 score
def f1_score_metric(preds, mask):
    preds = torch.argmax(preds[0], axis=1).view(-1).cpu().numpy()
    mask = torch.argmax(mask, axis=1).view(-1).cpu().numpy()
    f1 = f1_score(mask, preds, average='macro')
    return f1

# Function to compute mIoU
def compute_iou(pred, mask):
    if isinstance(pred, list):
        pred = torch.argmax(pred[0], dim=1).cpu().numpy().flatten()
    else:
        pred = torch.argmax(pred, dim=1).cpu().numpy().flatten()
    mask = torch.argmax(mask, dim=1).cpu().numpy().flatten()
    cm = confusion_matrix(mask, pred)
    intersection = np.diag(cm)
    union = np.sum(cm, axis=0) + np.sum(cm, axis=1) - np.diag(cm)
    iou = np.mean(intersection / (union + 1e-10))
    return iou

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

# Code reference: https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
def train(dataloader, model, loss_fn, optimizer, success_metric):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(torch.float32), y.to(torch.float32)
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        
        loss = 0
        if isinstance(pred, (list, tuple)):
            # print(type(pred))
            for i in range(len(pred)):
                # print(pred[i].shape)
                # print(len(pred))
                # print(y.shape)
                loss += loss_fn(pred[i], y)
        else:
            loss = loss_fn(pred, y)
            
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 10 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            accuracy = success_metric(pred, y)
            f1 = f1_score_metric(pred, y)
            iou = compute_iou(pred, y)
            print(f"loss: {loss:>7f}  accuracy: {accuracy:>7f}  F1: {f1:>7f}  mIoU: {iou:>7f}  [{current:>5d}/{size:>5d}]")
            
    # from https://github.com/Eladamar/fast_scnn/blob/master/main.py
    torch.save(model.state_dict(), '/Users/justindiamond/Desktop/EE562FinalProject-main/code/checkpoints/' + model.__class__.__name__ + datetime.datetime.today().strftime("%m_%d"))

def test(dataloader, model, loss_fn, success_metric):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(torch.float32), y.to(torch.float32)
            X, y = X.to(device), y.to(device)
            pred = model(X)
            if isinstance(pred, (list, tuple)):
                for i in range(len(pred)):
                    test_loss += loss_fn(pred[i], y).item()
    test_loss /= num_batches
    # score = success_metric(pred, y)
    # loss, current = loss.item(), (batch + 1) * len(X)
    accuracy = success_metric(pred, y)
    f1 = f1_score_metric(pred, y)
    iou = compute_iou(pred, y)
    print(f"loss: {test_loss:>7f}  accuracy: {accuracy:>7f}  F1: {f1:>7f}  mIoU: {iou:>7f}")
    # print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n score: {score:>7f} \n")

# code heavily borrowed from https://github.com/Eladamar/fast_scnn/blob/master/main.py
epochs = 1 # changed epochs from 100 to 10 for time
batch_size = 16 # increased batch size (images are 256x256 instead of 6000x6000)
learning_rate = 1e-3
device = 'mps'

train_image_path = '/Users/justindiamond/Desktop/ee562-transformer-main/data/Potsdam_6k/training/training/imgs'
train_label_path = '/Users/justindiamond/Desktop/ee562-transformer-main/data/Potsdam_6k/training/training/masks'
test_image_path = '/Users/justindiamond/Desktop/ee562-transformer-main/data/Potsdam_6k/validation/validation/imgs'
test_label_path = 'Users/justindiamond/Desktop/ee562-transformer-main/data/Potsdam_6k/validation/validation/masks'

ds_train = PotsdamDataset(train_image_path, train_label_path)
ds_test = PotsdamDataset(test_image_path, test_label_path)

dl_train = DataLoader(ds_train, batch_size, shuffle=True)
dl_test = DataLoader(ds_test, batch_size, shuffle=True)

# Code reference: https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
# model = FastSCNN(num_classes=6)

model = swinT(nclass=6, pretrained=None, aux=True, head="mlphead", edge_aux=False)
model_state_dict = torch.load('/Users/justindiamond/Desktop/EE562FinalProject-main/code//checkpoints/SwinT12_09',map_location=torch.device('cpu'))
model.load_state_dict(model_state_dict)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = CrossEntropyLoss()
success_metric = pixel_accuracy

model.to(device)
for epoch in range(epochs):
    # if epoch % 10 == 0:
    print('epoch', epoch+1, 'of', epochs)
    # train(dl_train, model, loss_fn, optimizer, success_metric)
    test(dl_test, model, loss_fn, success_metric)

import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import torchvision.transforms.functional as F
import numpy as np
import datetime

from dataset import PotsdamDataset
from models.FastSCNN import FastSCNN
from models.swinT import swin_tiny as swinT
from models.resunet import ResUNet

# Taken from https://github.com/Eladamar/fast_scnn/blob/master/metrics.py
def pixel_accuracy(preds, mask):
    if len(preds.shape) > 4:
        preds = torch.sum(*[preds[i] for i in range(len(preds))])
        
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
    model.to("cuda")
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        
        loss = 0
        if isinstance(pred, (list, tuple)):
            for i in range(len(pred)):
                loss += loss_fn(pred[i], y)
        else:
            loss = loss_fn(pred, y)
                
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 10 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            score = success_metric(pred, y)
            print(f"loss: {loss:>7f}  score: {score:>7f}  [{current:>5d}/{size:>5d}]")
            
    # from https://github.com/Eladamar/fast_scnn/blob/master/main.py
    torch.save(model.state_dict(), 'checkpoints/' + model.__class__.__name__ + datetime.datetime.today().strftime("%m_%d_%h"))

def test(dataloader, model, loss_fn, success_metric):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            if isinstance(pred, (list, tuple)):
                for i in range(len(pred)):
                    test_loss += loss_fn(pred[i], y).item()
    test_loss /= num_batches
    score = success_metric(pred, y)
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n score: {score:>7f} \n")

# code heavily borrowed from https://github.com/Eladamar/fast_scnn/blob/master/main.py
epochs = 1 # changed epochs from 100 to 10 for time
batch_size = 2 # increased batch size (images are 256x256 instead of 6000x6000)
learning_rate = 1e-3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache()

train_image_path = '../../data/Potsdam_6k/training/imgs'
train_label_path = '../../data/Potsdam_6k/training/masks'
test_image_path = '../../data/Potsdam_6k/validation/imgs'
test_label_path = '../../data/Potsdam_6k/validation/masks'

ds_train = PotsdamDataset(train_image_path, train_label_path)
ds_test = PotsdamDataset(test_image_path, test_label_path)

dl_train = DataLoader(ds_train, batch_size, shuffle=True)
dl_test = DataLoader(ds_test, batch_size, shuffle=True)

# Code reference: https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
#model = FastSCNN(num_classes=6)
#model = swinT(nclass=6, pretrained=True, aux=True, head="mlphead", edge_aux=False)
model = ResUNet()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = CrossEntropyLoss()
success_metric = pixel_accuracy

model.double()
model.to(device)
for epoch in range(epochs):
    if epoch % 10 == 0:
        print('epoch', epoch+1, 'of', epochs)
        train(dl_train, model, loss_fn, optimizer, success_metric)
        test(dl_test, model, loss_fn, success_metric)


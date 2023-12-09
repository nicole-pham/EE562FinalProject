# Code by Nicole

import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import datetime

from dataset import PotsdamDataset
from models.FastSCNN import FastSCNN
from models.swinT import swin_tiny as swinT
from models.resunet import ResUNet
from models.UNet_NN import UNet

# Code adapted from: https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
def train(dataloader, model, loss_fn, optimizer):
    '''
    Function for training our models. Saves the model periodically.

    INPUTS:
    dataloader: The torch dataloader that feeds in your images and masks
    model: The model to use for image segmentation
    loss_fn: The loss function to use to update the weights
    optimizer: The optimizer to use to determine the weight step
    '''
    size = len(dataloader.dataset)
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        
        loss = 0
        if isinstance(pred, (list, tuple)): # some models use auxilary predictions
            for i in range(len(pred)): # thus, we use all heads to calculate the loss
                loss += loss_fn(pred[i], y) # we do this by summing the losses
        else:
            loss = loss_fn(pred, y)
                
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 10 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
        
        if batch % 1000 == 0:
            torch.save(model.state_dict(), 'checkpoints/INTERMEDIATE' + str(batch) + model.__class__.__name__ + datetime.datetime.today().strftime("%m_%d_%H"))
            
    # this line was adapted from https://github.com/Eladamar/fast_scnn/blob/master/main.py
    torch.save(model.state_dict(), 'checkpoints/' + model.__class__.__name__ + datetime.datetime.today().strftime("%m_%d_%H"))

# code reference from https://github.com/Eladamar/fast_scnn/blob/master/main.py
epochs = 10 # max epochs to run, may stop the model sooner if/when we run out of time
batch_size = 3 # max batch size changes based on model
learning_rate = 1e-3
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.empty_cache() # sometimes pytorch doesn't do clear cache between runs, this is a failsafe

# Path to your images. Can use paths relative to your working dir or absolute paths
train_image_path = '../../data/Potsdam_6k/training/imgs'

ds_train = PotsdamDataset(train_image_path) # do transform the training data
dl_train = DataLoader(ds_train, batch_size, shuffle=True) # Train images in random order 

# Leave the model you want to test uncommented
#model = FastSCNN(num_classes=6)
#model = swinT(nclass=6, pretrained=True, aux=True, head="mlphead", edge_aux=False)
model = ResUNet()
#model = UNet()

# Code reference: https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # adam optimizer is pretty standard nowadays
loss_fn = CrossEntropyLoss() # all models to use cross entropy loss as their loss function

model.double()
model.to(device)
for epoch in range(epochs):
    print('epoch', epoch+1, 'of', epochs)
    train(dl_train, model, loss_fn, optimizer)


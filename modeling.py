import os.path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from model_struct import NNColabFiltering


def prep_dataloaders(X_train, y_train, X_val, y_val, batch_size):
    """This function converts the training and testing data set into tensor dataloaders for modeling"""
    # Convert training and test data to TensorDatasets
    trainset = TensorDataset(torch.from_numpy(np.array(X_train)).long(),
                             torch.from_numpy(np.array(y_train)).float())
    valset = TensorDataset(torch.from_numpy(np.array(X_val)).long(),
                           torch.from_numpy(np.array(y_val)).float())

    # Create Dataloaders for our training and test data to allow us to iterate over minibatches
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(valset, batch_size=batch_size, shuffle=False)

    return trainset, valset, trainloader, valloader


def train_model(model, criterion, optimizer, dataloaders, num_epochs=50, scheduler=None):
    """This function trains the model with the given parameters"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)  # Send model to GPU if available

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0

            # Get the inputs and labels, and send to GPU if available
            for (inputs, labels) in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the weight gradients
                optimizer.zero_grad()

                # Forward pass to get outputs and calculate loss
                # Track gradient only for training data
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model.forward(inputs).view(-1)
                    loss = criterion(outputs, labels)

                    # Backpropagation to get the gradients with respect to each weight
                    # Only if in train
                    if phase == 'train':
                        loss.backward()
                        # Update the weights
                        optimizer.step()

                # Convert loss into a scalar and add it to running_loss
                running_loss += np.sqrt(loss.item()) * labels.size(0)

            # Step along learning rate scheduler when in train
            if (phase == 'train') and (scheduler is not None):
                scheduler.step()

            # Calculate and display average loss and accuracy for the epoch
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            print('{} loss: {:.4f}'.format(phase, epoch_loss))


def launch_training(model_path, data):
    """This function prepares all the data for training and specify all the parameters for training the model.
    It also saves the model to the designated path after training"""
    X = data.loc[:, ['userIdInt', 'gameTitleInt']]
    y = data.loc[:, ['hours']]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=0)
    max_hours = data.max()['hours']
    batchsize = 64
    trainset, valset, trainloader, valloader = prep_dataloaders(X_train, y_train, X_val, y_val, batchsize)
    dataloaders = {'train': trainloader, 'val': valloader}
    n_users = X.loc[:, 'userIdInt'].max() + 1
    n_items = X.loc[:, 'gameTitleInt'].max() + 1
    model = NNColabFiltering(n_users, n_items, embedding_dim_users=40, embedding_dim_items=40, n_activations=50,
                             rating_range=[0., max_hours])
    criterion = nn.MSELoss()
    lr = 0.001
    n_epochs = 50
    wd = 0.6
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    train_model(model, criterion, optimizer, dataloaders, n_epochs, scheduler=None)
    torch.save(model, model_path)


if __name__ == "__main__":
    if not os.path.exists('temp/data.csv'):
        raise RuntimeError('Have you ran setup.py to create temp/ and required files?')
    data = pd.read_csv('temp/data.csv')
    launch_training('models/collab.pth', data)

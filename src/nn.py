#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 00:42:54 2021

@author: danish
"""

import time
import torch
import torch.nn as nn
from utils import get_device, print_inline, save_network
from data import get_data_loaders
import numpy as np

class RecommendationEngine(nn.Module):
    def __init__(self, unique_usr, unique_mov, emb_size):
        super(RecommendationEngine, self).__init__()
        #user embeding
        self.usr_embd = nn.Embedding(num_embeddings=unique_usr+1, 
                                     embedding_dim=emb_size)
        self.usr_bias = nn.Embedding(num_embeddings=unique_usr+1, 
                                     embedding_dim=1)
        #mov embed
        self.mov_embd = nn.Embedding(num_embeddings=unique_mov+1, 
                                     embedding_dim=emb_size)
        self.mov_bias = nn.Embedding(num_embeddings=unique_mov+1, 
                                     embedding_dim=1)
        #fc layer
        self.fc_layer = nn.Linear(in_features=16384, out_features=1)
        

    def forward(self, x, y):
        #x:user, y:movies
        prod_usr_mov = torch.inner(self.usr_embd(x), self.mov_embd(y))
        sum_usr_mov_bias = prod_usr_mov + self.usr_bias(x) + self.mov_bias(y)
        flatten_sum = torch.flatten(sum_usr_mov_bias, start_dim=1)
        output = torch.sigmoid(self.fc_layer(flatten_sum))
        return output
    
    
def get_criterions(device):
    loss = torch.nn.BCELoss().to(device)
    return loss

def get_optimizers(model, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, 
                                 betas=(0.5, 0.999))
    return optimizer


def train(train_data, valid_data, batch_size, lr, epochs, counts, early_stop):
    device = get_device()
    #data unpacking
    X_train, y_train = train_data
    X_valid, y_valid = valid_data
    unique_usr, unique_mov = counts
    #defining parameters and other variables
    emb_size = 50
    min_loss = 10e5
    es_counter = 0
    #model instation
    model = RecommendationEngine(unique_usr, unique_mov, 
                                 emb_size=emb_size).to(device)
    #criterion and optimizer
    criterion = get_criterions(device)
    optimizer = get_optimizers(model, lr)
    #dataloader
    train_loader = get_data_loaders(X_train, y_train, batch_size)
    valid_loader = get_data_loaders(X_valid, y_valid, batch_size)
    
    #train loop
    num_batches = len(X_train)//batch_size - 1
    history = {'train':[], 'validation':[]}
    print('\nStarting training!\n')
    for epoch in range(epochs):
        start = time.time()
        batch_loss = {'train':[], 'validation':[]}
        for i, (train_usr, train_mov, train_y) in enumerate(train_loader):
            if i==num_batches:
                break
            train_usr = train_usr.to(device)
            train_mov = train_mov.to(device)
            train_y = train_y.to(device)
            # forward pass
            model.zero_grad()
            Y_hat = model(train_usr, train_mov)
            loss_train = criterion(Y_hat, train_y)
            #backward pass
            loss_train.backward()
            optimizer.step()
            
            batch_loss['train'].append(loss_train.item())
            #info
            epoch_info = f"Epoch: {epoch+1}/{epochs} | Batch: {i}/{num_batches} | "
            train_loss_info = "Training Loss: " \
                              "{0:.4f} | ".format(loss_train.item())
            print_inline(epoch_info+train_loss_info)
        epoch_train_loss = np.mean(batch_loss['train'])
        history['train'].append(epoch_train_loss)
        #validation
        for i, (valid_usr, valid_mov, valid_y) in enumerate(valid_loader):
            #validation data
            valid_usr = valid_usr.to(device)
            valid_mov = valid_mov.to(device)
            valid_y = valid_y.to(device)
            #forward pass
            with torch.no_grad():
                Y_pred = model(valid_usr, valid_mov)
                loss_valid = criterion(Y_pred, valid_y)
            batch_loss['validation'].append(loss_valid.item())
        epoch_valid_loss = np.mean(batch_loss['validation'])
        history['validation'].append(epoch_valid_loss)
        end = int(time.time() - start)
        print("\nEpoch Training loss: {0:.4f} |".format(epoch_train_loss),
              "Validation loss: {0:.4f} |".format(epoch_valid_loss),
              f'Time taken in seconds: {end}.')
        if epoch_valid_loss<min_loss:
            es_counter = 0
            print('Loss Improved from {0:.4f} to'.format(min_loss),
                  '{0:.4f}, Saving the model.'.format(epoch_valid_loss))
            min_loss = epoch_valid_loss
            save_network(model, path='dump/model.pth')
        else:
            es_counter += 1
        if es_counter==early_stop:
            print('\nLoss did not improved since last {es_counter} epochs,',
                  'So terminating training!')
        print('')
    print('\nTraining Completed!')
    return model, history
    
    
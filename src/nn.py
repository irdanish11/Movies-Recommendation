#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 00:42:54 2021

@author: danish
"""

import torch
import torch.nn as nn

class RecommendationEngine(nn.Module):
    def __init__(self, unique_usr, emb_size):
        super(RecommendationEngine, self).__init__()
        #user embeding
        self.usr_embd = nn.Embedding(num_embeddings=unique_usr+1, 
                                     embedding_dim=emb_size)
        self.usr_bias = nn.Embedding(num_embeddings=unique_usr+1, 
                                     embedding_dim=emb_size)
        #mov embed
        self.mov_embd = nn.Embedding(num_embeddings=unique_usr+1, 
                                     embedding_dim=emb_size)
        self.mov_bias = nn.Embedding(num_embeddings=unique_usr+1, 
                                     embedding_dim=emb_size)
        #fc layer
        self.fc_layer = nn.Linear(in_features=50, out_features=1)
        

    def forward(self, x, y):
        #x:user, y:movies
        prod_usr_mov = torch.dot(self.usr_embd(x), self.mov_embd(y))
        sum_usr_mov_bias = prod_usr_mov + self.usr_bias(x) + self.mov_bias(y)
        flatten_sum = torch.flatten(sum_usr_mov_bias, start_dim=1)
        output = nn.ReLU(self.fc_layer(flatten_sum))
        return output
    
    
def get_criterions(device):
    loss = torch.nn.MSELoss().to(device)
    return loss

def get_optimizers(model, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
    return optimizer
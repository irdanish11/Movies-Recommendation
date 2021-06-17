#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 14:32:50 2021

@author: danish
"""

import pickle
import sys
import os
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.parallel
import numpy as np
import torchvision.utils as vutils
from tqdm import tqdm
from glob import glob


def get_device(cuda=True):
    cuda = cuda and torch.cuda.is_available()
    print("PyTorch version: {}".format(torch.__version__))
    if cuda:
        print("CUDA version: {}\n".format(torch.version.cuda))
    seed = np.random.randint(1, 10000)
    print("Random Seed: ", seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
    cudnn.benchmark = True
    device = torch.device("cuda:0" if cuda else "cpu")
    print('Device: ', device)
    return device 


def write_pickle(obj, path):
    with open(path, 'wb') as file:
        pickle.dump(obj, file)
   
        
def read_pickle(path):
    with open(path, 'rb') as file:
        obj = pickle.load(file)
    return obj


def save_network(net, path):
    torch.save(net.state_dict(), path)
    
    
def load_network(path, net):
    net.load_state_dict(torch.load(path))
    return net

def print_inline(string):
    sys.stdout.write('\r'+string)
    sys.stdout.flush() 
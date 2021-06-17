#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 12:40:23 2021

@author: danish
"""

from data import read_data, process_data, get_train_test_data
import os
from nn import train



if __name__=='__main__':
    data_path = '../dataset/data/ml-25m'
    dump_path = 'dump'
    os.makedirs(dump_path, exist_ok=True)
    movies, ratings = read_data(data_path)
    
    
    df = ratings.merge(movies, how='left', on='movieId')
    #processing df
    df, unique_mov, unique_usr = process_data(df, dump_path)
    
    X_train, X_test, y_train, y_test = get_train_test_data(df, dump_path, 
                                                           test_size=0.3) 
    train_data = (X_train, y_train)
    valid_data = (X_test, y_test)
    counts = (len(unique_usr), len(unique_mov))
    
    model, history = train(train_data, valid_data, batch_size=128, lr=0.001, epochs=1, 
                  counts=counts, early_stop=3)
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 01:29:32 2021

@author: danish
"""

from sklearn.model_selection import train_test_split
import pandas as pd
import os
import torch
from sklearn.preprocessing import LabelEncoder
from utils import write_pickle
from tqdm import tqdm

class DataGenerator(torch.utils.data.Dataset):
  def __init__(self, X, y):
        'Initialization'
        self.X = X
        self.y = y.reshape(-1, 1)
        self.users = X[:, 0].reshape(-1, 1)
        self.movies = X[:, 1].reshape(-1, 1)


  def __len__(self):
        'Denotes the total number of samples'
        return len(self.X)

  def __getitem__(self, index):
        'Generates one sample of data'    
    	# retrieve selected images
        usr = torch.tensor(self.users[index])
        mov = torch.tensor(self.movies[index])
        targets = torch.tensor(self.y[index], dtype=torch.float32)
        return usr, mov, targets

def get_data_loaders(X, y, batch_size, shuffle=True):
    data_gen = DataGenerator(X, y)
    #train_data, n_samples, patch_shape, transform
    dataloader = torch.utils.data.DataLoader(data_gen, batch_size, shuffle=shuffle)
    return dataloader

def read_data(data_path):
    print('\nReading data!')
    movies =  pd.read_csv(os.path.join(data_path, 'movies.csv'))
    ratings = pd.read_csv(os.path.join(data_path, 'ratings.csv'))
    return movies, ratings


def store_mov_name_genere(df, all_movs, dump_path):
    print('\nGenerating movie id to name and genre mappings, it takes a while!')
    id2name = {}
    for m in tqdm(all_movs):
        temp = df[df.movieId==m]
        id2name[m] = {'title':temp.title.unique()[0],
                      'genres':temp.genres.unique()[0]}
    write_pickle(id2name, path=os.path.join(dump_path, 'id2name.pkl'))
    print('\nMovie id to name, genre completed!')


def process_data(df, dump_path):
    #all movies
    all_movs = df.movieId.unique().tolist()
    #all users
    all_users = df.userId.unique().tolist()
    store_mov_name_genere(df, all_movs, dump_path)
    print('\nProcessing data!')
    #label encoding 
    le_movies = LabelEncoder()
    df['movieId'] = le_movies.fit_transform(df.movieId.values)
    le_user = LabelEncoder()
    df['userId'] = le_user.fit_transform(df.userId.values)
    unique_users = df.userId.unique()
    unique_movies = df.movieId.unique()
    #writing label encoders
    write_pickle(le_movies, path=os.path.join(dump_path, 'mov_encoder.pkl'))
    write_pickle(le_user, path=os.path.join(dump_path, 'usr_encoder.pkl'))
    write_pickle(all_movs, path=os.path.join(dump_path, 'all_movies.pkl'))
    write_pickle(all_users, path=os.path.join(dump_path, 'all_users.pkl'))
    return df, unique_movies, unique_users


def normalize_val(x, min_rating, max_rating):
    return (x - min_rating) / (max_rating - min_rating)

def get_train_test_data(df, dump_path, test_size=0.3):
    min_rating = min(df.rating)
    max_rating = max(df.rating)
    X = df[["userId", "movieId"]].values
    #print('\nNormalizing inputs')
    #sc = MinMaxScaler()
    #X = sc.fit_transform(X)
    # Normalize the targets between 0 and 1. 
    print('\nNormalizing targets')
    Y = df["rating"].apply(normalize_val, args=(min_rating, max_rating)).values
    print('\nPerforming train test split')
    X_train, X_test, y_train, y_test = train_test_split(X,Y, random_state=42, 
                                                        test_size=test_size, 
                                                        stratify=df.rating)
    #write_pickle(sc, path=os.path.join(dump_path, 'minmax_scaler.pkl'))
    return X_train, X_test, y_train, y_test
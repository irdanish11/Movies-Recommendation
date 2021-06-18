#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 12:40:23 2021

@author: danish
"""


import os
from nn import load_model_weights
from utils import read_pickle
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from data import get_data_loaders




def encode_movies(movs_unseen, dump_path):
    mov_encoder = read_pickle(os.path.join(dump_path, 'mov_encoder.pkl'))
    enc_unseen = []
    notfound = []
    for m in tqdm(movs_unseen):
        try:
            enc_unseen.append(mov_encoder.transform([m])[0])
        except ValueError:
            notfound.append(m)
    return np.array(enc_unseen), notfound

def prepare_data(user_csv_path, batch_size, dump_path):
    print('\nPreparing inputs\n')
    usr_encoder = read_pickle(os.path.join(dump_path, 'usr_encoder.pkl'))
    all_movs = read_pickle(os.path.join(dump_path, 'all_movies.pkl'))
    #reading data and extracting user id
    user_df = pd.read_csv(user_csv_path)
    user_id = user_df.userId.unique()[0]
    user_df = user_df[user_df.userId==user_id]
    #movies seen by user
    movs_seen = user_df.movieId.unique().tolist()
    #movies not seen by user
    movs_unseen = list(set(all_movs) - set(movs_seen))
    #encode unseen movies
    enc_unseen_mov, notfound = encode_movies(movs_unseen, dump_path)
    #encode user
    enc_user = usr_encoder.transform([user_id])[0]
    enc_users = np.array([enc_user]*len(movs_unseen))
    X_test = np.vstack((enc_users, enc_unseen_mov)).T
    num_batches = len(X_test)//batch_size - 1
    dummy_y = np.zeros(len(X_test))
    #getting test loader
    test_loader = get_data_loaders(X_test, dummy_y, batch_size, shuffle=False)
    return test_loader, num_batches, user_id

def decode_movies(dump_path, top_ind):
    mov_encoder = read_pickle(os.path.join(dump_path, 'mov_encoder.pkl'))
    movie_ids = []
    for i in top_ind:
        movie_ids.append(mov_encoder.inverse_transform([i])[0])
    return movie_ids


def get_predictions(test_loader, model, num_batches, device):
    print('\nMaking predictions using model!')
    predictions = []
    for i, (test_usr, test_mov, _) in tqdm(enumerate(test_loader)):
        if i==num_batches:
            break
        test_usr = test_usr.to(device)
        test_mov = test_mov.to(device)
        #get predictions
        with torch.no_grad():
            preds = model(test_usr, test_mov).cpu().numpy()
            for pred in preds:
                predictions.append(pred[0])
    sorted_preds = np.argsort(predictions) 
    return sorted_preds


def show_recommendations(user_id, movie_ids, id2name):
    print('\n======================================================='
          + '===============================')
    print(f'\t\t\tRecommendation for user: {user_id}')
    print('________________________________________________________'
          + '______________________________')
    print('\nSuggestions: Top 10\n')
    print('--------------------------------------------------------'
          + '------------------------------')
    print ("{:<55} {:<30} ".format('Titles','Genres'))
    print('--------------------------------------------------------'
          + '------------------------------')
    for mv_id in movie_ids:
        name = id2name[mv_id]['title']
        genre = id2name[mv_id]['genres']
        print ("{:<55} {:<30} ".format(name,genre))
    print('======================================================='
          + '===============================\n')


def suggest(user_csv_path, dump_path):
    #load model
    model, device = load_model_weights(dump_path, emb_size=50)
    
    test_loader, num_batches, user_id = prepare_data(user_csv_path, 
                                                     batch_size=128,
                                                     dump_path=dump_path)
    #getting predictions
    sorted_preds = get_predictions(test_loader, model, num_batches, device)                           
    top_ind = sorted_preds[0:10]
    #decode movies
    movie_ids = decode_movies(dump_path, top_ind)
    #loading movie id to name and genre mapping
    id2name = read_pickle(os.path.join(dump_path, 'id2name.pkl'))
    #show recommendations in tabular form
    show_recommendations(user_id, movie_ids, id2name)
    
 


if __name__=='__main__':
    dump_path = 'trained_model'
    user_csv_path = '../dataset/test_user.csv'
    #suggest movies
    suggest(user_csv_path, dump_path)


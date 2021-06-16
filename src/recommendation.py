#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 12:40:23 2021

@author: danish
"""

import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder


def read_data(data_path):
    movies =  pd.read_csv(os.path.join(data_path, 'movies.csv'))
    ratings = pd.read_csv(os.path.join(data_path, 'ratings.csv'))
    return movies, ratings


def process_data(df):
    #label encoding 
    le_movies = LabelEncoder()
    df['movieId'] = le_movies.fit_transform(df.movieId.values)
    le_user = LabelEncoder()
    df['userId'] = le_user.fit_transform(df.userId.values)
    unique_users = df.userId.unique()
    unique_movies = df.movieId.unique()
    return df, unique_movies, unique_users

if __name__=='__main__':
    data_path = '../dataset/data/ml-25m'
    movies, ratings = read_data(data_path)
    
    
    df = ratings.merge(movies, how='left', on='movieId')
        
    df.groupby(["movieId", "title", "genres"]).agg({
        "rating": "mean", "userId": "count"}).rename(
            columns={"userId": "n_reviews"}).sort_values("rating", 
                                                         ascending=False)
                                                         
    df_top_review = df.groupby(["movieId", "title", "genres"]).agg({
        "rating": "mean", "userId": "count"}).rename(
            columns={"userId": "n_reviews"}) 
    df_top_review = df_top_review[df_top_review.n_reviews > 10] 
    df_top_review.sort_values("rating", ascending=False)[:10]

    df_top_genres = df.groupby(["genres"]).agg({
        "rating": "mean", "userId": "count"}).rename(
            columns={"userId": "n_reviews"}) 
    df_top_genres = df_top_genres[df_top_genres.n_reviews > 10] 
    df_top_genres.sort_values("rating", ascending=False)[:10]
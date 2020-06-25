# -*- coding: utf-8 -*-
import os
import gc
from tqdm import tqdm

import pickle

from processing.process_numpy import get_zero_array
from processing.process_numpy import save_numpy

from utils.similarity import calculate_cosine_similarity
from utils.rating import mf_rating

from implicit.als import AlternatingLeastSquares

class MatrixFactorization:
    def __init__(self, name):
        self.name = name

        # number of data
        self.n_train = 0
        self.n_test = 0

        # data
        self.pt_train = None
        self.ps_train = None
        self.pt_test = None
        self.ps_test = None

        self.transformer_tag = None
        self.transformer_song = None

        # item-by-item similarity
        self.als_tag = None
        self.als_song = None

    def initialize(self, n_train, n_test, pt_train, ps_train, pt_test, ps_test, transformer_tag, transformer_song):
        self.n_train = n_train
        self.n_test = n_test

        self.pt_train = pt_train
        self.ps_train = ps_train
        self.pt_test = pt_test
        self.ps_test = ps_test

        self.transformer_tag = transformer_tag
        self.transformer_song = transformer_song

    def train(self):
        dir_name = './checkpoints/submission-val'
        file_name = 'als_tag-factor100-iter10.pkl'
        file_path = os.path.join(dir_name, file_name)
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                self.als_tag = pickle.load(f)
        else:
            self.als_tag = AlternatingLeastSquares(factors=100, regularization=0.08, iterations=10, calculate_training_loss=True)
            self.als_tag.fit(self.pt_train.T)
            
            with open(file_path, 'wb') as f:
                pickle.dump(self.als_tag, f)

        file_name = 'als_song-factor300-iter10.pkl'
        file_path = os.path.join(dir_name, file_name)
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                self.als_song = pickle.load(f)
        else:
            self.als_song = AlternatingLeastSquares(factors=300, regularization=0.08, iterations=30, calculate_training_loss=True)
            self.als_song.fit(self.ps_train.T)
            
            with open(file_path, 'wb') as f:
                pickle.dump(self.als_song, f)


    def predict(self, pid):
        rt = mf_rating(      
            pid,          
            self.pt_test, 
            self.als_tag, 
            self.transformer_tag.idf_
        )

        rs = mf_rating(
            pid, 
            self.ps_test, 
            self.als_song, 
            self.transformer_song.idf_
        )

        return rt, rs
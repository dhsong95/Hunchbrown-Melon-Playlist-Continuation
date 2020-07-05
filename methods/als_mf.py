# -*- coding: utf-8 -*-
"""
Author: DH Song
Last Modified: 2020.06.25
"""

import os
import pickle

import numpy as np
from implicit.als import AlternatingLeastSquares

from processing.process_sparse_matrix import write_sparse_matrix
from processing.process_sparse_matrix import load_sparse_matrix
from processing.process_sparse_matrix import horizontal_stack
from processing.process_sparse_matrix import vertical_stack

from similarity.cosine_similarity import calculate_cosine_similarity

from methods.method import Method

class ALSMFMethod(Method):
    """
    Matrix Factorization based method

    Args: 
    Return:
    """    
    def __init__(self, name, params):
        super().__init__(name)

        # Hyper parameter
        self.params = params

        # ALS Model
        self.model_tag = None
        self.model_song = None

    def _rate(self, pid, mode):
        '''
            rate each playlist.
            for the item in playlist.

        Args:
            pid(int): playlist id in test data
            mode(str): determine which item. tags or songs
        Return:
            rating(numpy array): playlist and [tags or songs] rating 
        '''        
        assert mode in ['tags', 'songs']

        model = self.model_tag if mode == 'tags' else self.model_song

        pid = self.n_train + pid
        rating = np.dot(model.user_factors[pid, :], model.item_factors.T).reshape(-1)

        return rating

    def initialize(self, n_train, n_test, pt_train, ps_train, pt_test, ps_test, transformer_tag, transformer_song):
        '''
            initialize necessary variables

        Args: 
            n_train(int): number of train data
            n_test(int): number of test data
            pt_train(scipy csr matrix): playlist-tag sparse matrix from train data
            ps_train(scipy csr matrix): playlist-song sparse matrix from train data
            pt_test(scipy csr matrix): playlist-tag sparse matrix from test data
            ps_test(scipy csr matrix): playlist-song sparse matrix from test data
            transformer_tag(sci-kit learn TfIdfTransformer model): tag TfIdfTransformer model
            transformer_song(sci-kit learn TfIdfTransformer model): song TfIdfTransformer model
        Return:
        '''
        super().initialize(n_train, n_test, pt_train, ps_train, pt_test, ps_test, transformer_tag, transformer_song)

        self.model_tag = AlternatingLeastSquares(factors=self.params['tag']['factors'], 
                                                 regularization=self.params['tag']['regularization'],
                                                 iterations=self.params['tag']['iterations'],
                                                 calculate_training_loss=True,
                                                 use_gpu=True)
        self.model_song = AlternatingLeastSquares(factors=self.params['song']['factors'], 
                                                  regularization=self.params['song']['regularization'],
                                                  iterations=self.params['song']['iterations'],
                                                  calculate_training_loss=True,
                                                  use_gpu=True)


    def train(self, checkpoint_dir='./checkpoints'):
        '''
            train MF Method.
            ALS Model fitting
            Save ALS Model

        Args: 
            checkpoint_dir(str): where to save model
        Return:
        '''
        dirname = os.path.join(checkpoint_dir, self.name)
        if not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)

        filename = os.path.join(dirname, 'als-mf-tag.pkl')
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                self.model_tag = pickle.load(f)
        else:
            data = vertical_stack(self.pt_train, self.pt_test)
            data = (data * self.params['tag']['confidence']).astype('double')
            self.model_tag.fit(data.T)            
            with open(filename, 'wb') as f:
                pickle.dump(self.model_tag, f)

        filename = os.path.join(dirname, 'als-mf-song.pkl')
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                self.model_song = pickle.load(f)
        else:
            data = vertical_stack(self.ps_train, self.ps_test)
            data = (data * self.params['song']['confidence']).astype('double')
            self.model_song.fit(data.T)
            with open(filename, 'wb') as f:
                pickle.dump(self.model_song, f)

    def predict(self, pid):
        '''
            rating the playlist

        Args: 
            pid(int): playlist id
        Return:
            rating_tag(numpy array): playlist id and tag rating
            rating_song(numpy array): playlist id and song rating
        '''
        rating_tag = np.zeros(self.n_tag)
        rating_song = self._rate(pid, mode='songs')

        return rating_tag, rating_song
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

from similarity.cosine_similarity import calculate_cosine_similarity

from methods.method import Method

class MFMethod(Method):
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

        n = self.n_tag if mode == 'tags' else self.n_song
        test = self.pt_test if mode == 'tags' else self.ps_test
        model = self.model_tag if mode == 'tags' else self.model_song
        idf = self.transformer_tag.idf_ if mode == 'tags' else self.transformer_song.idf_

        rating = np.zeros(n)
        item_features = model.item_factors
        playlist_feature = np.zeros(item_features.shape[1])

        if test[pid, :].count_nonzero() != 0:
            denominator = 0.0
            for item in test[pid, :].nonzero()[1]:
                playlist_feature += (idf[item] * item_features[item])
                denominator += idf[item]
            playlist_feature /= denominator
        
            rating = np.dot(playlist_feature, item_features.T)
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
                                                 calculate_training_loss=True)
        self.model_song = AlternatingLeastSquares(factors=self.params['song']['factors'], 
                                                  regularization=self.params['song']['regularization'],
                                                  iterations=self.params['song']['iterations'],
                                                  calculate_training_loss=True)


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

        filename = os.path.join(dirname, 'lmf-tag.pkl')
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                self.model_tag = pickle.load(f)
        else:
            self.model_tag.fit(self.pt_train.T)            
            with open(filename, 'wb') as f:
                pickle.dump(self.model_tag, f)

        filename = os.path.join(dirname, 'lmf-song.pkl')
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                self.model_song = pickle.load(f)
        else:
            self.model_song.fit(self.ps_train.T)
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
        rating_tag = self._rate(pid, mode='tags') 
        rating_song = self._rate(pid, mode='songs')

        return rating_tag, rating_song
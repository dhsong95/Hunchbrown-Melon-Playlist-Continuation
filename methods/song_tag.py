# -*- coding: utf-8 -*-
"""
Author: DH Song
Last Modified: 2020.06.25
"""

import os

import numpy as np

from processing.process_sparse_matrix import write_sparse_matrix
from processing.process_sparse_matrix import load_sparse_matrix
from processing.process_sparse_matrix import horizontal_stack
from processing.process_sparse_matrix import transform_idf

from similarity.cosine_similarity import calculate_cosine_similarity

from methods.method import Method

class SongTagCrossMethod(Method):
    """
    KNN based on IDF Transformed sparse matrix

    Args: 
    Return:
    """    
    def __init__(self, name, k_ratio=0.001):
        super().__init__(name)

        # tag-by-song
        self.ts_matrix = None

        self.ts_matrix_idf = None
        self.st_matrix_idf = None

    def _rate(self, pid, mode):
        '''
            rate each playlist.
            for the item in playlist. calculate consider only the k nearest neighbors.

        Args:
            pid(int): playlist id in test data
            mode(str): determine which item. tags or songs
        Return:
            rating(numpy array): playlist and [tags or songs] rating 
        '''        
        assert mode in ['tags', 'songs']

        n = self.n_tag if mode == 'tags' else self.n_song
        test_cross = self.ps_test if mode == 'tags' else self.pt_test
        cross_matrix = self.st_matrix_idf if mode == 'tags' else self.ts_matrix_idf

        rating = np.zeros(n)
        for idx in test_cross[pid, :].nonzero()[1]:
            rating += (cross_matrix[test_cross[pid, idx]]).toarray().reshape(-1)
            
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

    def train(self, checkpoint_dir='./checkpoints'):
        '''
            train the IDF KNN Method.
            find k nearest neighbors based on playlist to playlist similarity
            Save the similarity matrix

        Args: 
            checkpoint_dir(str): where to save similarity matrix
        Return:
        '''
        print(self.ts_matrix.shape)
        _, self.ts_matrix_idf = transform_idf(self.ts_matrix)
        _, self.st_matrix_idf = transform_idf(self.ts_matrix.T)

    def predict(self, pid, mode):
        '''
            rating the playlist

        Args: 
            pid(int): playlist id
        Return:
            rating_tag(numpy array): playlist id and tag rating
            rating_song(numpy array): playlist id and song rating
        '''
        rating = self._rate(pid, mode=mode)

        return rating
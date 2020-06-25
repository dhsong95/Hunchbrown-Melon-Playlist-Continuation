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

from similarity.cosine_similarity import calculate_cosine_similarity

from methods.method import Method

class IdfKNNMethod(Method):
    """
    KNN based on IDF Transformed sparse matrix

    Args: 
    Return:
    """    
    def __init__(self, name, k_ratio=0.001):
        super().__init__(name)

        # Hyper Parameter
        self.k_ratio = k_ratio
        self.k = 0

        # test-by-train similarity (p:playlist)
        self.pp_similarity = None
        self.neighbors = None

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
        train = self.pt_train if mode == 'tags' else self.ps_train
        test = self.pt_test if mode == 'tags' else self.ps_test
        similarity = self.pp_similarity
        idf = self.transformer_tag.idf_ if mode == 'tags' else self.transformer_song.idf_

        rating = np.zeros(n)
        for neighbor in self.pp_similarity[pid, :].toarray().argsort(axis=-1)[:, ::-1][0, :self.k]:
            rating += (similarity[pid, neighbor] * train[neighbor, :]).toarray().reshape(-1)  
        
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

        k = int(self.k_ratio * self.n_train)
        # make k to be multiply of 10
        self.k = k + (10 - (k % 10))
        print('KNN works with {} neighbor'.format(self.k))

    def train(self, checkpoint_dir='./checkpoints'):
        '''
            train the IDF KNN Method.
            find k nearest neighbors based on playlist to playlist similarity
            Save the similarity matrix

        Args: 
            checkpoint_dir(str): where to save similarity matrix
        Return:
        '''
        pt_idf_train = self.transformer_tag.transform(self.pt_train)
        ps_idf_train = self.transformer_song.transform(self.ps_train)
        pt_idf_test = self.transformer_tag.transform(self.pt_test)
        ps_idf_test = self.transformer_song.transform(self.ps_test)

        dirname = os.path.join(checkpoint_dir, self.name)
        if not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)

        filename = os.path.join(dirname, 'playlist-similarity.npz')
        if os.path.exists(filename):
            self.pp_similarity = load_sparse_matrix(filename)
        else:
            self.pp_similarity = calculate_cosine_similarity(
                # Determine Playlist Similarity more emphasis on song feature 
                horizontal_stack(pt_idf_test, ps_idf_test, [0.15, 0.85]),
                horizontal_stack(pt_idf_train, ps_idf_train, [0.15, 0.85])
            )
            write_sparse_matrix(self.pp_similarity, filename)

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
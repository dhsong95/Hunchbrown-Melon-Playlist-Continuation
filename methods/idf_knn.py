# -*- coding: utf-8 -*-
""" Idf Knn Method class.

Author: Hunchbrown - DH Song
Last Modified: 2020.07.14

Idf KNN Method class for Playlist continuation task.
"""

import os

import numpy as np

from methods.method import Method

from processing.process_sparse_matrix import horizontal_stack
from processing.process_sparse_matrix import load_sparse_matrix
from processing.process_sparse_matrix import write_sparse_matrix

from similarity.cosine_similarity import calculate_cosine_similarity

class IdfKNNMethod(Method):
    """ Idf Knn Method class for playlist continuation task.
    
    KNN based model with IDF Transformed sparse matrix.

    Attributes:
        name (str)  : name of method
        k_ratio (float)    : portion of k
        k (int) : number of neighbors to be considered base on k_ratio
        pp_similarity (csr_matrix)  : playlist in test to playlist in train cosine similarity
        neighbors (csr_matrix)  : sorted index of playlist based on similarity
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
        """ Make ratings.
        
        Rate on items(tag/song) based on test data, which index is pid.
        
        Args:
            pid (int)   : playlist id in test data
            mode (str)  : determine which item. tags or songs
        Return:
            rating(numpy array): playlist and [tags or songs] rating 
        """ 

        assert mode in ['tags', 'songs']

        n = self.n_tag if mode == 'tags' else self.n_song           # number of item in total
        train = self.pt_train if mode == 'tags' else self.ps_train 
        similarity = self.pp_similarity

        rating = np.zeros(n)
        for neighbor in similarity[pid, :].toarray().argsort(axis=-1)[:, ::-1][0, :self.k]:
            rating += (similarity[pid, neighbor] * train[neighbor, :]).toarray().reshape(-1)  
        
        return rating

    def initialize(self, n_train, n_test, pt_train, ps_train, pt_test, ps_test, transformer_tag, transformer_song):
        """ initialize necessary variables for Method.

        initialize necessary data structure.

        Args: 
            n_train (int)   : number of playlist in train dataset.
            n_test (int)    : number of playlist in test dataset. 
            pt_train (csr_matrix)   : playlist to tag sparse matrix made from train dataset.
            ps_train (csr_matrix)   : playlist to tag sparse matrix made from train dataset.
            pt_test (csr_matrix)    : playlist to tag sparse matrix made from test dataset.
            ps_test (csr_matrix)    : playlist to song sparse matrix made from test dataset.
            transformer_tag (TfidfTransformer)  : scikit-learn TfidfTransformer model fitting pt_train.
            transformer_song (TfidfTransformer) : scikit-learn TfidfTransformer model fitting ps_train.
        Return:
        """    

        super().initialize(n_train, n_test, pt_train, ps_train, pt_test, ps_test, transformer_tag, transformer_song)

        k = int(self.k_ratio * self.n_train)
        # make k to be multiply of 10
        self.k = k + (10 - (k % 10))
        print('\tKNN works with {} neighbor...'.format(self.k))

    def train(self, checkpoint_dir='./checkpoints'):
        """ Train Idf KNN Method

        find k nearest neighbors based on playlist in test to playlist in train similarity
        Save the similarity matrix

        Args: 
            checkpoint_dir (str)    : where to save similarity matrix.
        Return:
        """

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
        """ Make ratings

        rate the playlist, which index in test sparse matrix is pid.

        Args: 
            pid(int)    : playlist id in test sparse matrix
        Return:
            rating_tag(ndarray) : playlist id and tag rating
            rating_song(ndarray): playlist id and song rating
        """

        rating_tag = self._rate(pid, mode='tags')
        rating_song = self._rate(pid, mode='songs')

        return rating_tag, rating_song
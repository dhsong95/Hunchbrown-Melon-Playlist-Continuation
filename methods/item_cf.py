# -*- coding: utf-8 -*-
""" Idf Knn Method class.

Author: Hunchbrown - DH Song
Last Modified: 2020.07.14

Item CF Method class for Playlist continuation task.
"""

import os

import numpy as np

from methods.method import Method

from processing.process_sparse_matrix import load_sparse_matrix
from processing.process_sparse_matrix import write_sparse_matrix

from similarity.cosine_similarity import calculate_cosine_similarity

class ItemCFMethod(Method):
    """ Item Collaborative Filterint Method class for playlist continuation task.
    
    Item Collaborative Filtering Method.

    Attributes:
        name (str)  : name of method
        tt_similarity (csr_matrix)  : all tag by tag cosine similarity matrix
        ss_similarity (csr_matrix)  : all song by song cosine similarity matrix
    Return:
    """    

    def __init__(self, name):
        super().__init__(name)

        # item-by-item similarity (t:tag, s:song)
        self.tt_similarity = None
        self.ss_similarity = None

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

        n = self.n_tag if mode == 'tags' else self.n_song
        test = self.pt_test if mode == 'tags' else self.ps_test
        similarity = self.tt_similarity if mode == 'tags' else self.ss_similarity
        idf = self.transformer_tag.idf_ if mode == 'tags' else self.transformer_song.idf_

        rating = np.zeros(n)
        for item in test[pid, :].nonzero()[1]:
            rating += (similarity[item, :] * idf[item]).toarray().reshape(-1)

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

    def train(self, checkpoint_dir='./checkpoints'):
        """ Train Item CF Method

        Calculate the tag-tag similarity and song-song similarity.
        Save the similarity matrix

        Args: 
            checkpoint_dir (str)    : where to save similarity matrix.
        Return:
        """

        dirname = os.path.join(checkpoint_dir, self.name)
        if not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)

        filename = os.path.join(dirname, 'tag-similarity.npz')
        if os.path.exists(filename):
            self.tt_similarity = load_sparse_matrix(filename)
        else:
            pt_idf_train = self.transformer_tag.transform(self.pt_train)
            self.tt_similarity = calculate_cosine_similarity(pt_idf_train.T)
            write_sparse_matrix(self.tt_similarity, filename)

        filename = os.path.join(dirname, 'song-similarity.npz')
        if os.path.exists(filename):
            self.ss_similarity = load_sparse_matrix(filename)
        else:
            ps_idf_train = self.transformer_song.transform(self.ps_train)
            self.ss_similarity = calculate_cosine_similarity(ps_idf_train.T)
            write_sparse_matrix(self.ss_similarity, filename)

    def predict(self, pid):
        """ Make ratings

        rate the playlist, which index in test sparse matrix is pid.

        Args: 
            pid(int)    : playlist id in test sparse matrix
        Return:
            rating_tag(ndarray) : playlist id and tag rating
            rating_song(ndarray): playlist id and song rating
        """

        rating_tag = self._rate(pid, 'tags')
        rating_song = self._rate(pid, 'songs')

        return rating_tag, rating_song
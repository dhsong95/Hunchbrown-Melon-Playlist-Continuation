# -*- coding: utf-8 -*-
"""
Author: DH Song
Last Modified: 2020.06.25
"""

import os

import numpy as np

from processing.process_sparse_matrix import write_sparse_matrix
from processing.process_sparse_matrix import load_sparse_matrix

from similarity.cosine_similarity import calculate_cosine_similarity

from methods.method import Method

class ItemCFMethod(Method):
    """
    Item based Collaborative Filtering Method

    Args: 
    Return:
    """    
    def __init__(self, name):
        super().__init__(name)

        # item-by-item similarity (t:tag, s:song)
        self.tt_similarity = None
        self.ss_similarity = None

    def _rate(self, pid, mode):
        '''
            rate each playlist.
            for the item in playlist. calculate idf-weighted similary items.

        Args:
            pid(int): playlist id in test data
            mode(str): determine which item. tags or songs
        Return:
            rating(numpy array): playlist and [tags or songs] rating 
        '''
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
            train the Item Cf Method.
            Calculate the tag-tag similarity and song-song similarity.
            Save the similarity matrix

        Args: 
            checkpoint_dir(str): where to save similarity matrix
        Return:
        '''
        dirname = os.path.join(checkpoint_dir, self.name)
        if not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)

        filename = os.path.join(dirname, 'tag-similarity.npz')
        if os.path.exists(filename):
            self.tt_similarity = load_sparse_matrix(filename)
        else:
            self.tt_similarity = calculate_cosine_similarity(self.pt_train.T)
            write_sparse_matrix(self.tt_similarity, filename)

        filename = os.path.join(dirname, 'song-similarity.npz')
        if os.path.exists(filename):
            self.ss_similarity = load_sparse_matrix(filename)
        else:
            ps_idf_train = self.transformer_song.transform(self.ps_train)
            self.ss_similarity = calculate_cosine_similarity(ps_idf_train.T)
            write_sparse_matrix(self.ss_similarity, filename)

    def predict(self, pid):
        '''
            rating the playlist

        Args: 
            pid(int): playlist id
        Return:
            rating_tag(numpy array): playlist id and tag rating
            rating_song(numpy array): playlist id and song rating
        '''
        rating_tag = self._rate(pid, 'tags')
        rating_song = self._rate(pid, 'songs')

        return rating_tag, rating_song
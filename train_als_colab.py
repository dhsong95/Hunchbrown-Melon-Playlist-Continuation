# -*- coding: utf-8 -*-
"""
Author: DH Song
Last Modified: 2020.07.04
"""

from tqdm import tqdm

import numpy as np

import fire

from processing.process_json import load_json
from processing.process_json import write_json
from processing.process_dataframe import to_dataframe
from processing.process_dataframe import get_item_idx_dictionary
from processing.process_sparse_matrix import to_sparse_matrix
from processing.process_sparse_matrix import transform_idf


from methods.als_mf import ALSMFMethod

class ALSColab:
    """
    Playlist Continuation Class

    Args: 
    Return:
    """    
    def __init__(self):
        # number of data
        self.n_train = 0
        self.n_test = 0

        # item to index dictionary
        self.tag2idx = dict()
        self.song2idx = dict()
        self.playlist2idx = dict()

        # sparse matrix (p:playlist, t:tag, s:song)
        self.pt_train = None
        self.pt_test = None
        self.ps_train = None
        self.ps_test = None

        # sparse matrix - idf transformed
        self.pt_idf_train = None
        self.ps_idf_train = None

        # (tf)-idf transformer
        self.transformer_tag = None
        self.transformer_song = None

        # method and weight
        self.methods = list()
        self.weights = list()

    def _prepare_data(self, train, test):
        '''
            make data for playlist continuation task.
            preprae sparse matrix, item dictionary etc.

        Args:
            train(json): train data
            test(json): test data
        Return:
        '''        
        df_train = to_dataframe(train)
        df_test = to_dataframe(test)

        self.n_train = len(df_train)
        self.n_test = len(df_test)

        self.tag2idx = get_item_idx_dictionary(df_train, df_test, 'tags')
        self.song2idx = get_item_idx_dictionary(df_train, df_test, 'songs')
        self.playlist2idx = get_item_idx_dictionary(df_train, df_test, 'id')

        # pt: playlist-tag sparse matrix
        # ps: playlist-song sparse matrix
        self.pt_train = to_sparse_matrix(df_train, self.playlist2idx, self.tag2idx, 'tags')
        self.ps_train = to_sparse_matrix(df_train, self.playlist2idx, self.song2idx, 'songs')

        self.pt_test = to_sparse_matrix(df_test, self.playlist2idx, self.tag2idx, 'tags', correction=self.n_train)
        self.ps_test = to_sparse_matrix(df_test, self.playlist2idx, self.song2idx, 'songs', correction=self.n_train)

        print('IDF transformation...')
        self.transformer_tag, self.pt_idf_train = transform_idf(self.pt_train)
        self.transformer_song, self.ps_idf_train = transform_idf(self.ps_train)

        print('\n*********Train Data*********')
        print('Shape of Playlist-Tag Sparse Matrix: \t{}'.format(self.pt_train.shape))
        print('Shape of Playlist-Song Sparse Matrix: \t{}'.format(self.ps_train.shape))
        print('\n*********Test Data*********')
        print('Shape of Playlist-Tag Sparse Matrix: \t{}'.format(self.pt_test.shape))
        print('Shape of Playlist-Song Sparse Matrix: \t{}\n'.format(self.ps_test.shape))

    def _train_methods(self):
        '''
            for each method. train the method.
            calculate similarity, model fitting etc.

        Args:
        Return:
        '''        
        for method in self.methods:
            print('Method {} training...'.format(method.name))
            method.initialize(self.n_train, self.n_test, self.pt_train, self.ps_train, self.pt_test, self.ps_test, self.transformer_tag, self.transformer_song)
            method.train()

    def run(self, train_fname, test_fname):
        '''
            running playlist continuation task.

        Args:
            train_fname(str): train filename.
            test_fname(str): test filename.
        Return:
        '''        
        print("Loading train file...")
        train = load_json(train_fname)

        print("Loading test file...")
        test = load_json(test_fname)

        print('Preparing data...')
        self._prepare_data(train, test)

        print('Training methods...')
        # als_params = {
        #     'tag': {
        #         'factors': 256,
        #         'regularization': 0.001,
        #         'iterations': 500,
        #         'confidence': 100
        #     },
        #     'song': {
        #         'factors': 512,
        #         'regularization': 0.001,
        #         'iterations': 1000,
        #         'confidence': 100
        #     }
        # }
        als_params = {
            'tag': {
                'factors': 512,
                'regularization': 0.001,
                'iterations': 300,  # 250 -> 300
                'confidence': 100
            },
            'song': {
                'factors': 512,
                'regularization': 0.001,
                'iterations': 200,  # 250 -> 200
                'confidence': 100
            }
        }

        self.methods = [
            ALSMFMethod('als-matrix-factorization', params=als_params), 
        ]
        self.weights = [
            (1.0, 1.0),
        ]
        self._train_methods()

if __name__ == "__main__":
    fire.Fire(ALSColab)
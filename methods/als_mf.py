# -*- coding: utf-8 -*-
""" Idf Knn Method class.

Author: Hunchbrown - DH Song
Last Modified: 2020.07.14

ALS Matrix Factorization Method class for Playlist continuation task.
"""

import os
import pickle

import implicit
from implicit.als import AlternatingLeastSquares
import numpy as np

from methods.method import Method

from processing.process_sparse_matrix import horizontal_stack
from processing.process_sparse_matrix import load_sparse_matrix
from processing.process_sparse_matrix import vertical_stack
from processing.process_sparse_matrix import write_sparse_matrix

from similarity.cosine_similarity import calculate_cosine_similarity

class ALSMFMethod(Method):
    """ ALS Matrix Factorization Method class for playlist continuation task.
    
    ALS Matrix Factorization Method.

    Attributes:
        name (str)  : name of method
        params (dict)   : ALS model parameters
        model_tag (ALS Model)  : ALS Model for tag
        model_song (ALS Mode)  : ALS Model for song
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
        """ Make ratings.
        
        Rate on items(tag/song) based on test data, which index is pid.
        
        Args:
            pid (int)   : playlist id in test data
            mode (str)  : determine which item. tags or songs
        Return:
            rating(numpy array): playlist and [tags or songs] rating 
        """ 
        
        assert mode in ['tags', 'songs']

        model = self.model_tag if mode == 'tags' else self.model_song

        pid = self.n_train + pid
        rating = np.dot(model.user_factors[pid, :], model.item_factors.T).reshape(-1)

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

        self.model_tag = AlternatingLeastSquares(factors=self.params['tag']['factors'], 
                                                 regularization=self.params['tag']['regularization'],
                                                 iterations=self.params['tag']['iterations'],
                                                 calculate_training_loss=True,
                                                 use_gpu=implicit.cuda.HAS_CUDA)
        self.model_song = AlternatingLeastSquares(factors=self.params['song']['factors'], 
                                                  regularization=self.params['song']['regularization'],
                                                  iterations=self.params['song']['iterations'],
                                                  calculate_training_loss=True,
                                                  use_gpu=implicit.cuda.HAS_CUDA)


    def train(self, checkpoint_dir='./checkpoints'):
        """ Train ALS MF Method

        Fit ALS Model on train and test dataset.
        Save ALS Model.

        Args: 
            checkpoint_dir (str)    : where to save similarity matrix.
        Return:
        """

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
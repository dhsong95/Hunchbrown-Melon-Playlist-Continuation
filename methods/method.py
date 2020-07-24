# -*- coding: utf-8 -*-
""" Base Method class.

Author: Hunchbrown - DH Song
Last Modified: 2020.07.20

Base Method class for Playlist continuation task.
"""

class Method:
    """ Base Method class.

    Base Method class for Playlist continuation task.

    Attributes: 
        name (str)  : name of method
        n_train (int)   : number of playlist in train dataset.
        n_test (int)    : number of playlist in test dataset. 
        n_tag (int) : number of tag in train and test dataset.
        n_song (int)    : number of song in train and test dataset. 
        pt_train (csr_matrix)   : playlist to tag sparse matrix made from train dataset.
        pt_test (csr_matrix)    : playlist to tag sparse matrix made from test dataset.
        ps_train (csr_matrix)   : playlist to tag sparse matrix made from train dataset.
        ps_test (csr_matrix)    : playlist to song sparse matrix made from test dataset.
        transformer_tag (TfidfTransformer)  : scikit-learn TfidfTransformer model fitting pt_train.
        transformer_song (TfidfTransformer) : scikit-learn TfidfTransformer model fitting ps_train.
    Return:
    """    

    def __init__(self, name):
        # name of method
        self.name = name

        # number of data
        self.n_train = 0
        self.n_test = 0

        # number of item
        self.n_tag = 0
        self.n_song = 0

        # sparse matrix (p:playlist, t:tag, s:song)
        self.pt_train = None
        self.ps_train = None
        self.pt_test = None
        self.ps_test = None

        # (tf)-idf transformer
        self.transformer_tag = None
        self.transformer_song = None

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

        self.n_train = n_train
        self.n_test = n_test

        self.pt_train = pt_train
        self.ps_train = ps_train
        self.pt_test = pt_test
        self.ps_test = ps_test

        self.transformer_tag = transformer_tag
        self.transformer_song = transformer_song

        self.n_tag = self.pt_train.shape[1]
        self.n_song = self.ps_train.shape[1]


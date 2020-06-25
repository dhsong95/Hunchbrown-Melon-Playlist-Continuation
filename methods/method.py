# -*- coding: utf-8 -*-
"""
Author: DH Song
Last Modified: 2020.06.25
"""

class Method:
    """
    Super class for methods

    Args: 
        name(str): name of method
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
        """
        initialize necessary variables for method

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


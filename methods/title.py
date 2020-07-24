# -*- coding: utf-8 -*-
""" Title Method class.

Author: Hunchbrown - DH Song
Last Modified: 2020.07.20

Title Method class for Playlist continuation task, especially for cold start problem.
"""

import numpy as np
from scipy.sparse import csr_matrix

from khaiii import KhaiiiApi
from khaiii import KhaiiiExcept

from methods.method import Method

from processing.process_sparse_matrix import transform_idf

class TitleMethod(Method):
    """ Title KNN Method class for playlist continuation task cold start problem.
    
    Title KNN Method.

    Attributes:
        name (str)  : name of method
        playlist2idx (dict) : playlist to index dictionary.
        title2playlist (dict)   : title to list of playlists dictionary.
        token2idx (dict)    : NLP processed token to index dictionary.
        token2title (dict)  : NLP processed token to list of titles dictionary. 
        doc2vec_model (doc2vec) : Doc2Vec Model in gensim.
        tt_matrix (sparse matirx)   : NLP processed token to tag matrix
        ts_matirx (sparse matrix)   : NLP processed token to song matrix
        api (KhaiiApi)  : Korean Tokenizer
    Return:
    """    

    def __init__(self, name):
        super().__init__(name)

        self.playlist2idx = dict()
        self.title2playlist = dict()
        self.token2idx = dict()
        self.token2title = dict()

        self.unique_token = set()

        self.doc2vec_model = None

        self.tt_matrix = None
        self.ts_matrix = None

        self.api = KhaiiiApi()


    def _tokenize_title(self, title):
        """ Tokenize playlist title.
        
        Tokenize playlist title using khaiii.

        Attributes:
            title (str) : playlist title
        Return:
            token (list)   : list of "lexicon/tag" token
        """    

        token = list()
        try:
            words = self.api.analyze(title)
        except KhaiiiExcept:
            words = list()

        for word in words:
            for morph in word.morphs:
                if morph.tag[:2] in ['NN', 'VV', 'VA', 'VC', 'MM', 'XR'] or morph.tag == 'MAG':
                    token.append('/'.join([morph.lex, morph.tag]))
            
        return token

    def _prepare_data(self):
        """ Prepare necessary data structures for Title KNN Method.

        Prepare necessary data structures for Title KNN Method.

        """    

        ### tokenize using khaiii
        ### make csr matrix (token - tag | song)
        row = {'tag': list(), 'song': list()}
        col = {'tag': list(), 'song': list()}
        data = {'tag': list(), 'song': list()}

        token_id = 0
        for title, playlist in self.title2playlist.items():

            # check wheter this title is in train dataset (not validation or test dataset)
            has_train_playlist = False
            for p in playlist:
                playlist_id = self.playlist2idx[p]
                if playlist_id < self.n_train:
                    has_train_playlist = True
                    break
            
            if not has_train_playlist:
                continue

            token = self._tokenize_title(title)

            for t in token:
                if t in self.token2idx:
                    token_id = self.token2idx[t]
                else:
                    self.token2idx[t] = token_id

                for p in playlist:
                    playlist_id = self.playlist2idx[p]
                    if playlist_id < self.n_train:
                        for item_id in self.pt_train[playlist_id].nonzero()[1]:
                            row['tag'].append(token_id)
                            col['tag'].append(item_id)
                            data['tag'].append(1)

                        for item_id in self.ps_train[playlist_id].nonzero()[1]:
                            row['song'].append(token_id)
                            col['song'].append(item_id)
                            data['song'].append(1)

                token_id = len(self.token2idx)

        self.tt_matrix = csr_matrix((data['tag'], (row['tag'], col['tag'])), dtype=float)
        self.ts_matrix = csr_matrix((data['song'], (row['song'], col['song'])), dtype=float)

        _, self.tt_matrix = transform_idf(self.tt_matrix)
        _, self.ts_matrix = transform_idf(self.ts_matrix)

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

        title_matrix = self.tt_matrix if mode == 'tags' else self.ts_matrix
        n = self.n_tag if mode == 'tags' else self.n_song

        idx2playlist = {idx:playlist for playlist, idx in self.playlist2idx.items()}
        playlist2title = dict()
        for title, playlists in self.title2playlist.items():
            for playlist in playlists:
                playlist2title[playlist] = title

        rating = np.zeros(n)

        playlist = idx2playlist[pid + self.n_train]
        title = playlist2title[playlist]
        token = self._tokenize_title(title)
        token = [t for t in token if t in self.token2idx.keys()]
        token_ids = [self.token2idx[t] for t in token]

        if len(token_ids) == 0:
            return rating

        rating = np.sum(title_matrix[token_ids, :].toarray(), axis=0).reshape(-1)
        return rating

    def initialize(self, n_train, n_test, pt_train, ps_train, pt_test, ps_test, transformer_tag, transformer_song, checkpoint_dir='./checkpoints'):
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
            checkpoint_dir (str)    : where to save similarity matrix.
        Return:
        """    

        super().initialize(n_train, n_test, pt_train, ps_train, pt_test, ps_test, transformer_tag, transformer_song)

        ### tokenize using khaiii
        ### make csr matrix (token - tag | song)
        row = {'tag': list(), 'song': list()}
        col = {'tag': list(), 'song': list()}
        data = {'tag': list(), 'song': list()}

        token_id = 0
        for title, playlist in self.title2playlist.items():

            # check wheter this title is in train dataset (not validation or test dataset)
            has_train_playlist = False
            for p in playlist:
                playlist_id = self.playlist2idx[p]
                if playlist_id < self.n_train:
                    has_train_playlist = True
                    break
            
            if not has_train_playlist:
                continue

            token = self._tokenize_title(title)

            for t in token:
                if t in self.token2idx:
                    token_id = self.token2idx[t]
                else:
                    self.token2idx[t] = token_id

                for p in playlist:
                    playlist_id = self.playlist2idx[p]
                    if playlist_id < self.n_train:
                        for item_id in self.pt_train[playlist_id].nonzero()[1]:
                            row['tag'].append(token_id)
                            col['tag'].append(item_id)
                            data['tag'].append(1)

                        for item_id in self.ps_train[playlist_id].nonzero()[1]:
                            row['song'].append(token_id)
                            col['song'].append(item_id)
                            data['song'].append(1)

                token_id = len(self.token2idx)

        self.tt_matrix = csr_matrix((data['tag'], (row['tag'], col['tag'])), dtype=float)
        self.ts_matrix = csr_matrix((data['song'], (row['song'], col['song'])), dtype=float)

        _, self.tt_matrix = transform_idf(self.tt_matrix)
        _, self.ts_matrix = transform_idf(self.ts_matrix)

    def predict(self, pid, mode):
        """ Make ratings based on mode.

        rate the playlist, which index in test sparse matrix is pid based on mode.

        Args: 
            pid (int)   : playlist id in test sparse matrix
            mode (str)  : tags or songs
        Return:
            rating (ndarray)    : playlist id and rating
        """
        rating = self._rate(pid, mode=mode) 
        return rating
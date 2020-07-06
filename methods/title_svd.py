# -*- coding: utf-8 -*-
"""
Author: DH Song
Last Modified: 2020.06.25
"""

import os
import pickle
from tqdm import tqdm
from collections import namedtuple

import numpy as np
from scipy.sparse import csr_matrix
from gensim.models import doc2vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.decomposition import TruncatedSVD

from khaiii import KhaiiiApi
from khaiii import KhaiiiExcept

from processing.process_sparse_matrix import write_sparse_matrix
from processing.process_sparse_matrix import load_sparse_matrix
from processing.process_sparse_matrix import horizontal_stack
from processing.process_sparse_matrix import vertical_stack

from similarity.cosine_similarity import calculate_cosine_similarity

from methods.method import Method

class TitleSVDMethod(Method):
    """
    Truncated SVD with using Name

    Args: 
    Return:
    """    
    def __init__(self, name):
        super().__init__(name)

        self.playlist2idx = dict()
        self.title2playlist = dict()
        self.token2idx = dict()
        self.token2title = dict()

        self.doc2vec_model = None

        self.tt_matrix = None
        self.ts_matrix = None

    def _prepare_data(self):
        ### tokenize using khaiii
        ### make csr matrix (token - tag | song)
        api = KhaiiiApi()
        row = {'tag': list(), 'song': list()}
        col = {'tag': list(), 'song': list()}
        data = {'tag': list(), 'song': list()}
        rid = 0
        for title, playlist in tqdm(self.title2playlist.items()):
            tokens = list()
            try:
                words = api.analyze(title)
            except KhaiiiExcept:
                words = list()
                tokens = ['/ZZ']

            for word in words:
                for morph in word.morphs:
                    tokens.append('/'.join([morph.lex, morph.tag]))
            tokens = ' '.join(tokens)
            if tokens in self.token2idx:
                rid = self.token2idx[tokens]
            else:
                self.token2title[tokens] = list()
                self.token2idx[tokens] = rid

            self.token2title[tokens].append(title)

            for p in playlist:
                idx = self.playlist2idx[p]
                if idx < self.n_train:
                    for cid in self.pt_train[idx].nonzero()[1]:
                        row['tag'].append(rid)
                        col['tag'].append(cid)
                        data['tag'].append(1)

                    for cid in self.ps_train[idx].nonzero()[1]:
                        row['song'].append(rid)
                        col['song'].append(cid)
                        data['song'].append(1)
            rid = len(self.token2idx)

        self.tt_matrix = csr_matrix((data['tag'], (row['tag'], col['tag'])), shape=(len(self.token2idx), self.n_tag))
        self.ts_matrix = csr_matrix((data['song'], (row['song'], col['song'])), shape=(len(self.token2idx), self.n_song))

        # for a in range(1000, 1010):
        #     print(self.ts_matrix[a].nonzero())
        #     idx2token = {idx:token for token, idx in self.token2idx.items()}
        #     token = idx2token[a]
        #     for title in self.token2title[token]:
        #         for playlist in self.title2playlist[title]:
        #             idx = self.playlist2idx[playlist]
        #             if idx < self.ps_train.shape[0]:
        #                 print(self.ps_train[idx].nonzero())

        #     print(idx2token[a])

    def _rate(self, pid, mode):
        '''
            rate each playlist.
            for the item in playlist.

        Args:
            pid(int): playlist id in test data
            mode(str): determine which item. tags or songs
        Return:
            rating(numpy array): playlist and [tags or songs] rating 
        '''        
        assert mode in ['tags', 'songs']
        title_matrix = self.tt_matrix if mode == 'tags' else self.ts_matrix
        n = self.n_tag if mode == 'tags' else self.n_song

        idx2playlist = {idx:playlist for playlist, idx in self.playlist2idx.items()}
        playlist2title = dict()
        for title, playlists in self.title2playlist.items():
            for playlist in playlists:
                playlist2title[playlist] = title

        rating = np.zeros(n)

        playlist = idx2playlist[pid]
        title = playlist2title[playlist]

        api = KhaiiiApi()
        token = list()
        for word in api.analyze(title):
            for morph in word.morphs:
                token.append('/'.join([morph.lex, morph.tag]))

        doc_vec = self.doc2vec_model.infer_vector(token)
        counter = 0
        for tag, similarity in self.doc2vec_model.docvecs.most_similar([doc_vec], topn=len(self.token2idx)):
            if counter >= 100 or similarity < 0.5:
                break

            if title_matrix[tag].count_nonzero() > 0:
                rating += (title_matrix[tag].toarray() * similarity).reshape(-1)
                counter += 1

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
            train MF Method.
            ALS Model fitting
            Save ALS Model

        Args: 
            checkpoint_dir(str): where to save model
        Return:
        '''

        self._prepare_data()

        dirname = os.path.join(checkpoint_dir, self.name)
        if not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)

        filename = os.path.join(dirname, 'doc2vec.model')
        if os.path.exists(filename):
            self.doc2vec_model = doc2vec.Doc2Vec.load(filename)
        else:
            tagged_doc = [TaggedDocument(token.split(), [idx]) for token, idx in self.token2idx.items()]

            self.doc2vec_model = doc2vec.Doc2Vec(vector_size=500, alpha=0.05, min_alpha=0.05, seed=2020)
            self.doc2vec_model.build_vocab(tagged_doc)

            for _ in tqdm(range(50)):
                self.doc2vec_model.train(
                    tagged_doc, 
                    total_examples=self.doc2vec_model.corpus_count, 
                    epochs=self.doc2vec_model.epochs
                )
                self.doc2vec_model.alpha -= 0.002
                self.doc2vec_model.min_alpha = self.doc2vec_model.alpha

            self.doc2vec_model.save(filename)
        
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

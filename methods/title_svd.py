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
from sklearn.decomposition import TruncatedSVD

from khaiii import KhaiiiApi

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

        self.title2idx = dict()
        self.token2idx = dict()
        self.tkn2ttl = dict()

        self.doc_vectorizer = None

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
        for title, indices in tqdm(self.title2idx.items()):
            tokens = list()
            for word in api.analyze(title):
                for morph in word.morphs:
                    tokens.append('/'.join([morph.lex, morph.tag]))
            tokens = ' '.join(tokens)
            self.token2idx[tokens] = rid
            self.tkn2ttl[rid] = indices
            for idx in indices:
                if idx >= self.n_train:
                    continue

                for cid in self.pt_train[idx].nonzero()[1]:
                    row['tag'].append(rid)
                    col['tag'].append(cid)
                    data['tag'].append(1)

                for cid in self.ps_train[idx].nonzero()[1]:
                    row['song'].append(rid)
                    col['song'].append(cid)
                    data['song'].append(1)
            
            rid += 1

        self.tt_matrix = csr_matrix((data['tag'], (row['tag'], col['tag'])), shape=(len(self.token2idx), self.n_tag))
        self.ts_matrix = csr_matrix((data['song'], (row['song'], col['song'])), shape=(len(self.token2idx), self.n_song))

        for a in range(100, 110):
            print(self.tt_matrix[a].nonzero())
            print(self.tkn2ttl[a])
            for b in self.tkn2ttl[a]:
                if b < self.pt_train.shape[0]:
                    print(self.pt_train[b].nonzero())
                else:
                    print(self.pt_test[b - self.n_train].nonzero())

            idx2token = {idx:token for token, idx in self.token2idx.items()}
            print(idx2token[a])

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
        test = self.pt_test if mode == 'tags' else self.ps_test

        idx2title = {idx:title for title, idx in self.title2idx.items()}
        title = idx2title[pid + self.n_train]

        api = KhaiiiApi()

        token = api.analyze(title)
        doc_vec = self.doc_vectorizer.infer_vector(token)
        
        for t, similarity in self.doc_vectorizer.docvecs.most_similar([doc_vec]):
            pass

        return None

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
            self.doc_vectorizer = doc2vec.Doc2Vec.load(filename)
        else:
            TaggedDocument = namedtuple('TaggedDocument', 'words tag')
            tagged_doc = [TaggedDocument(token, [idx]) for token, idx in self.token2idx.items()]

            self.doc_vectorizer = doc2vec.Doc2Vec(size=500, alpha=0.025, min_alpha=0.025, seed=2020)
            self.doc_vectorizer.build_vocab(tagged_doc)

            for _ in tqdm(range(10)):
                self.doc_vectorizer.train(tagged_doc)
                self.doc_vectorizer.alpha -= 0.002
                self.doc_vectorizer.min_alpha = self.doc_vectorizer.alpha

            self.doc_vectorizer.save('doc2vec.model')

    def predict(self, pid):
        '''
            rating the playlist

        Args: 
            pid(int): playlist id
        Return:
            rating_tag(numpy array): playlist id and tag rating
            rating_song(numpy array): playlist id and song rating
        '''
        # rating_tag = self._rate(pid, mode='tags') 
        # rating_song = self._rate(pid, mode='songs')

        # return rating_tag, rating_song
        return None
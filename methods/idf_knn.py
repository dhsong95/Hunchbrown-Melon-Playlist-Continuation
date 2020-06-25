# -*- coding: utf-8 -*-
import os
import gc
from tqdm import tqdm

import numpy as np

from processing.process_sparse_matrix import horizontal_stack
from processing.process_numpy import get_zero_array
from processing.process_numpy import save_numpy
from utils.similarity import calculate_cosine_similarity
from utils.rating import idf_knn_rating


class IdfKnn:
    def __init__(self, name, k=10, idf=True):
        self.name = name

        # parameter
        self.k = k

        # number of data
        self.n_train = 0
        self.n_test = 0

        # data
        self.pt_train = None
        self.ps_train = None
        self.pt_test = None
        self.ps_test = None

        self.pt_train_idf = None
        self.ps_train_idf = None
        self.pt_test_idf = None
        self.ps_test_idf = None

        self.transformer_tag = None
        self.transformer_song = None

        # test-by-train similarity
        self.similarity = None

        self.neighbors = None

        self.ratings_tag = None
        self.ratings_song = None

    def initialize(self, n_train, n_test, pt_train, ps_train, pt_test, ps_test, transformer_tag, transformer_song):
        self.n_train = n_train
        self.n_test = n_test

        self.pt_train = pt_train
        self.pt_train_idf = transformer_tag.transform(pt_train)

        self.ps_train = ps_train
        self.ps_train_idf = transformer_song.transform(ps_train)

        self.pt_test = pt_test
        self.pt_test_idf = transformer_tag.transform(pt_test)

        self.ps_test = ps_test
        self.ps_test_idf = transformer_song.transform(ps_test)

        self.transformer_tag = transformer_tag
        self.transformer_song = transformer_song
        

    def train(self):
        
        self.similarity = calculate_cosine_similarity(
            horizontal_stack(self.pt_test_idf, self.ps_test_idf, [0.15, 0.85]),
            horizontal_stack(self.pt_train_idf, self.ps_train_idf, [0.15, 0.85])
        )
        self.neighbors = self.similarity.toarray().argsort(axis=-1)[:, ::-1][:, :self.k]

    def predict(self, pid):
        # tag_dir = './checkpoints/rating/{}/tag'.format(self.name)
        # song_dir = './checkpoints/rating/{}/song'.format(self.name)

        # if not os.path.exists(tag_dir):
        #     os.makedirs(tag_dir, exist_ok=True)
        # if not os.path.exists(song_dir):
        #     os.makedirs(song_dir, exist_ok=True)

        # assert self.pt_test.shape[0] == self.ps_test.shape[0]

        # for pid in tqdm(range(self.n_test)):
        #     if pid % 1000 == 0:
        #         start = pid
        #         end = pid + 999 if pid + 999 < self.n_test else self.n_test - 1
        #         ratings_tag = get_zero_array(shape=(end - start + 1, self.pt_test.shape[1]))
        #         ratings_song = get_zero_array(shape=(end - start + 1, self.ps_test.shape[1]))

        rt = idf_knn_rating(     
            pid,
            self.pt_train,
            self.pt_test,
            self.neighbors, 
            self.similarity
        )
            # ratings_tag[pid % 1000] = rt

        rs = idf_knn_rating(
            pid,
            self.ps_train,
            self.ps_test,
            self.neighbors,
            self.similarity
        )
            # ratings_song[pid % 1000] = rs

        #     if pid % 1000 == 999 or pid == self.n_test - 1:
        #         gc.collect()

        #         save_numpy(os.path.join(tag_dir, '{}-{}'.format(start, end)), ratings_tag)
        #         save_numpy(os.path.join(song_dir, '{}-{}'.format(start, end)), ratings_song)

        # return tag_dir, song_dir
        return rt, rs
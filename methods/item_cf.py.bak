# -*- coding: utf-8 -*-
import os
import gc
from tqdm import tqdm

from processing.process_numpy import get_zero_array
from processing.process_numpy import save_numpy

from utils.similarity import calculate_cosine_similarity
from utils.rating import item_cf_rating

import scipy.sparse as sp
class ItemCF:
    def __init__(self, name):
        self.name = name

        # number of data
        self.n_train = 0
        self.n_test = 0

        # data
        self.pt_train = None
        self.ps_train = None
        self.pt_test = None
        self.ps_test = None

        self.transformer_tag = None
        self.transformer_song = None

        # item-by-item similarity
        self.tt_similarity = None
        self.ss_similarity = None

    def initialize(self, n_train, n_test, pt_train, ps_train, pt_test, ps_test, transformer_tag, transformer_song):
        self.n_train = n_train
        self.n_test = n_test

        self.pt_train = pt_train
        self.ps_train = ps_train
        self.pt_test = pt_test
        self.ps_test = ps_test

        self.transformer_tag = transformer_tag
        self.transformer_song = transformer_song

    def train(self):
        ps_idf_train = self.transformer_song.transform(self.ps_train)

        self.tt_similarity = calculate_cosine_similarity(self.pt_train.T)
        self.ss_similarity = calculate_cosine_similarity(ps_idf_train.T)
        # self.tt_similarity = sp.load_npz('./temp/tag-similarity.npz')
        # self.ss_similarity = sp.load_npz('./temp/song-similarity.npz')

    def predict(self, pid):
        # tag_dir = './checkpoints/rating/{}/tag'.format(self.name)
        # song_dir = './checkpoints/rating/{}/song'.format(self.name)

        # if not os.path.exists(tag_dir):
        #     os.makedirs(tag_dir, exist_ok=True)
        # if not os.path.exists(song_dir):
        #     os.makedirs(song_dir, exist_ok=True)

        # for pid in tqdm(range(self.n_test)):
        #     if pid % 1000 == 0:
        #         start = pid
        #         end = pid + 999 if pid + 999 < self.n_test else self.n_test - 1
        #         ratings_tag = get_zero_array(shape=(end - start + 1, self.pt_test.shape[1]))
        #         ratings_song = get_zero_array(shape=(end - start + 1, self.ps_test.shape[1]))
        
        rt = item_cf_rating(      
            pid,          
            self.pt_test, 
            self.tt_similarity, 
            self.transformer_tag.idf_
        )
            # ratings_tag[pid % 1000] = rt

        rs = item_cf_rating(
            pid, 
            self.ps_test, 
            self.ss_similarity, 
            self.transformer_song.idf_
        )
            # ratings_song[pid % 1000] = rs

            # if pid % 1000 == 999 or pid == self.n_test - 1:
            #     gc.collect()

            #     save_numpy(os.path.join(tag_dir, '{}-{}'.format(start, end)), ratings_tag)
            #     save_numpy(os.path.join(song_dir, '{}-{}'.format(start, end)), ratings_song)

        return rt, rs
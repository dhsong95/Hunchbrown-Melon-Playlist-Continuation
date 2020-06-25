# -*- coding: utf-8 -*-
import os
import fire
from tqdm import tqdm

import numpy as np
from sklearn.preprocessing import MinMaxScaler

from processing.process_json import load_json
from processing.process_json import write_json

from processing.process_dataframe import to_dataframe
from processing.process_dataframe import get_item_idx_dictionary

from processing.process_sparse_matrix import to_sparse_matrix
from processing.process_sparse_matrix import transform_idf

from processing.process_numpy import get_zero_array
from processing.process_numpy import load_numpy

from methods.item_cf import ItemCF
from methods.idf_knn import IdfKnn
from methods.mf import MatrixFactorization


class PlaylistContinuation:
    def __init__(self):
        # Number of data
        self.n_train = 0
        self.n_test = 0

        # Item to Id Dictionary
        self.tag2idx = dict()
        self.song2idx = dict()
        self.playlist2idx = dict()

        self.pt_train = None
        self.ps_train = None
        self.pt_test = None
        self.ps_test = None

        self.transformer_tag = None
        self.transformer_song = None

        self.pt_idf_train = None
        self.ps_idf_train = None

        self.methods = list()

    
    def _prepare_data(self, train, test):
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

        print(self.pt_train.shape)
        print(self.ps_train.shape)

        self.pt_test = to_sparse_matrix(df_test, self.playlist2idx, self.tag2idx, 'tags', correction=self.n_train)
        self.ps_test = to_sparse_matrix(df_test, self.playlist2idx, self.song2idx, 'songs', correction=self.n_train)

        print(self.pt_test.shape)
        print(self.ps_test.shape)

        print('IDF transformation...')
        self.transformer_tag, self.pt_idf_train = transform_idf(self.pt_train)
        self.transformer_song, self.ps_idf_train = transform_idf(self.ps_train)

    def _train_methods(self):
        for method in self.methods:
            print('Method {} training...'.format(method.name))
            method.initialize(self.n_train, self.n_test, self.pt_train, self.ps_train, self.pt_test, self.ps_test, self.transformer_tag, self.transformer_song)
            method.train()

    def _select_items(self, rating, top_item, already_item, idx2item, n):
        idx = get_zero_array(shape=(n, ))
        counter = 0
        for item_id in rating.argsort()[::-1]:
            if rating[item_id] == 0 or counter == n:
                break
            if item_id not in already_item:
                idx[counter] = item_id
                counter += 1

        # filling                
        for item_id in top_item:
            if counter == n:
                break
            if item_id not in already_item and item_id not in idx:
                idx[counter] = item_id
                counter += 1

        return [idx2item[item_id] for item_id in idx]


    def _generate_answers(self):            
        idx2tag = {idx:tag for tag, idx in self.tag2idx.items()}
        idx2song = {idx:song for song, idx in self.song2idx.items()}
        idx2playlist = {idx-self.n_train:playlist for playlist, idx in self.playlist2idx.items() if idx >= self.n_train}

        answers = []

        print("Generating answers...")
        for pid in tqdm(range(self.n_test)):
            playlist = idx2playlist[pid]

            rating_tag = get_zero_array(shape=(len(self.tag2idx), ))
            rating_song = get_zero_array(shape=(len(self.song2idx), ))

            for method, weight in zip(self.methods, self.weights):
                rt, rs = method.predict(pid)

                r_min = rt.min()
                r_max = rt.max()

                if r_max != 0.0:
                    rt = (rt - r_min) / (r_max - r_min)

                r_min = rs.min()
                r_max = rs.max()

                if r_max != 0.0:
                    rs = (rs - r_min) / (r_max - r_min)

                rating_tag += (rt * weight[0]).reshape(-1)
                rating_song += (rs * weight[1]).reshape(-1)


            # if pid % 1000 == 0:
            #     start = pid
            #     end = pid + 999 if start + 999 < self.n_test else self.n_test - 1
            #     ratings_tag = get_zero_array(shape=(end - start + 1, len(self.tag2idx)))
            #     ratings_song = get_zero_array(shape=(end - start + 1, len(self.song2idx)))
            #     for rt_dir, rs_dir, weight in zip(rt_dirs, rs_dirs, weights):
            #         ratings_tag = load_numpy(os.path.join(rt_dir, '{}-{}.npy'.format(start, end)), weight[0])
            #         ratings_song = load_numpy(os.path.join(rs_dir, '{}-{}.npy'.format(start, end)), weight[1])

            top_tag = self.transformer_tag.idf_.argsort()
            already_tag = self.pt_test[pid, :].nonzero()[1]
            tags = self._select_items(rating_tag, top_tag, already_tag, idx2tag, 10)

            top_song = self.transformer_song.idf_.argsort()
            already_song = self.ps_test[pid, :].nonzero()[1]
            songs = self._select_items(rating_song, top_song, already_song, idx2song, 100)

            answers.append({
                "id": playlist,
                "songs": songs,
                "tags": tags,
            })

        return answers

    def run(self, train_fname, test_fname):
        print("Loading train file...")
        train = load_json(train_fname)

        print("Loading test file...")
        test = load_json(test_fname)

        print('Preparing data...')
        self._prepare_data(train, test)

        print('Training methods...')
        # self.methods = [ItemCF('item-cf'), MatrixFactorization('matrix-factorization'), IdfKnn('idf-knn', k=250)]
        self.methods = [IdfKnn('idf-knn', k=9)]
        # (Tag Wieght, Song Wieght)
        # self.weights = [(0.1, 0.1), (0.05, 0.2), (0.85, 0.7)]
        self.weights = [(1, 1)]

        # self.methods = [MatrixFactorization('matrix-factorization')]
        # (Tag Wieght, Song Wieght)
        # self.weights = [(1, 1)]

        self._train_methods()
        answers = self._generate_answers()

        print("Writing answers...")
        write_json(answers, "results/results.json")


if __name__ == "__main__":
    fire.Fire(PlaylistContinuation)

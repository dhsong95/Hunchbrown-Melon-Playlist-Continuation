# -*- coding: utf-8 -*-
"""
Author: DH Song
Last Modified: 2020.06.25
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


from methods.idf_knn import IdfKNNMethod
from methods.item_cf import ItemCFMethod
from methods.mf import MFMethod

class PlaylistContinuation:
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

    def _select_items(self, rating, top_item, already_item, idx2item, n):
        '''
            based on rating, recommend $n items.

        Args:
            rating(numpy array): array of rating on items(tags, songs).
            top_item(numpy array): most frequently appeared item in train data
            already_item(nuympy array): item in test data
            idx2item(dict): dictionary to recover original item name
            n(int): number of recommendation 
        Return:
        '''        
        idx = np.zeros(shape=(n))
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
        '''
            make rating and recommendations.

        Args:
        Return:
            answers(dict): recommendations
        '''        
        idx2tag = {idx:tag for tag, idx in self.tag2idx.items()}
        idx2song = {idx:song for song, idx in self.song2idx.items()}
        idx2playlist = {idx-self.n_train:playlist for playlist, idx in self.playlist2idx.items() if idx >= self.n_train}

        answers = []

        for pid in tqdm(range(self.n_test)):
            rating_tag = np.zeros(shape=(len(self.tag2idx)))
            rating_song = np.zeros(shape=(len(self.song2idx)))

            for method, weight in zip(self.methods, self.weights):
                weight_tag = weight[0]
                weight_song = weight[1]

                rt, rs = method.predict(pid)

                r_min = rt.min(-1)
                r_max = rt.max(-1)
                if r_max != 0:
                    rt = (rt - r_min) / (r_max - r_min)

                r_min = rs.min(-1)
                r_max = rs.max(-1)
                if r_max != 0:
                    rs = (rs - r_min) / (r_max - r_min)

                rating_tag += (rt * weight_tag)
                rating_song += (rs * weight_song)


            playlist = idx2playlist[pid]

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
        mf_params = {
            'tag': {
                'factors': 100,
                'regularization': 0.08,
                'iterations': 10
            },
            'song': {
                'factors': 300,
                'regularization': 0.08,
                'iterations': 30
            }

        }
        self.methods = [
            ItemCFMethod('item-collaborative-filtering'), 
            # MFMethod('matrix-factorization', params=mf_params), 
            # Trail 1 Failed
            # IdfKNNMethod('idf-knn', k_ratio=0.005)
            IdfKNNMethod('idf-knn', k_ratio=0.003)
        ]
        # (Tag Weight, Song Weight) per method
        # Trail 1 Failed
        # self.weights = [
        #     (0.55, 0.1), 
        #     # (0.01, 0.0),
        #     (0.45, 0.9), 
        # ]
        self.weights = [
            (0.6, 0.4), # (0.7, 0.4), # (0.5, 0.3), # (0.6, 0.15)
            # (0.01, 0.0),
            (0.4, 0.6) # (0.3, 0.6) # (0.5, 0.7), # (0.4, 0.85)
        ]
        self._train_methods()

        print('Generating answers...')
        answers = self._generate_answers()

        print("Writing answers...")
        write_json(answers, "results/results.json")

if __name__ == "__main__":
    fire.Fire(PlaylistContinuation)
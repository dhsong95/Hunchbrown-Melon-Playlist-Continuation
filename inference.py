# -*- coding: utf-8 -*-
""" Main playlist continuation system model. 

Author: Hunchbrown - DH Song
Last Modified: 2020.07.24

Make inference on songs/tags based on seed songs/tags in train and test dataset.
"""

import argparse

import numpy as np
from tqdm import tqdm

from processing.process_dataframe import get_item_idx_dictionary
from processing.process_dataframe import map_title_to_playlist
from processing.process_dataframe import to_dataframe
from processing.process_json import load_json
from processing.process_json import write_json
from processing.process_sparse_matrix import to_sparse_matrix
from processing.process_sparse_matrix import transform_idf

from methods.als_mf import ALSMFMethod
from methods.idf_knn import IdfKNNMethod
from methods.item_cf import ItemCFMethod
from methods.title import TitleMethod

from utils.normalize import normalize_zero_to_one

class PlaylistContinuation:
    """ Playlist continuastion model.

    Recommend songs/tags based on seeds songs/tags in train and test datasets.

    Atrtibutes:
        n_train (int)   : number of playlist in train dataset.
        n_test (int)    : number of playlist in test dataset. 
        tag2idx (dict)  : tag to index(starts from 0) dictionary.
        song2idx (dict) : song to index(starts from 0) dictionary.
        playlist2idx (dict) : playlist to index(starts from 0) dictionary.
        title2playlist (dict)   : title to list of playlists dictionary
        pt_train (csr_matrix)   : playlist to tag sparse matrix made from train dataset.
        pt_test (csr_matrix)    : playlist to tag sparse matrix made from test dataset.
        ps_train (csr_matrix)   : playlist to tag sparse matrix made from train dataset.
        ps_test (csr_matrix)    : playlist to song sparse matrix made from test dataset.
        pt_idf_train (csr_matirx)   : TF-IDF transformed pt_train.
        ps_idf_train (csr_matirx)   : TF-IDF transformed ps_train.
        transformer_tag (TfidfTransformer)  : scikit-learn TfidfTransformer model fitting pt_train.
        transformer_song (TfidfTransformer) : scikit-learn TfidfTransformer model fitting ps_train.
        methods (list)  : list of Method classes for playlist continuation.
        weights (list)  : list of (tag weight, song weight) to be multiplied to each Method in methods.
        title_method (TitleMethod)  : additional method for cold start problem, using title information.
    """    

    def __init__(self):
        # number of data
        self.n_train = 0
        self.n_test = 0

        # item to index dictionary
        self.tag2idx = dict()
        self.song2idx = dict()
        self.playlist2idx = dict()
        self.title2playlist = dict()

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
        self.title_method = TitleMethod(name='title')

    def _prepare_data(self, train, test):
        """ Prepare data necessary for task.

        Major data structures necessary for task.
            1. Sparse Matrix
                - playlist to tag/song sparse matrix made from train and test dataset.
            2. Dictionary
                - playlist/tag/song to index dictionary. index is used for sparse matrix row or column index.
                - playlist title to list of playlist dictionary. used for title method for cold start.
        
        Args:
            train (json): train data
            test (json) : test data
        Return:
        """        

        ### Convert JSON to pandas DataFrame
        df_train = to_dataframe(train)
        df_test = to_dataframe(test)

        self.n_train = len(df_train)
        self.n_test = len(df_test)

        ### Make dictionary
        self.tag2idx = get_item_idx_dictionary(df_train, mode='tags')
        self.song2idx = get_item_idx_dictionary(df_train, mode='songs')
        self.playlist2idx = get_item_idx_dictionary(df_train, df_test, 'id')

        self.title2playlist = map_title_to_playlist(df_train, df_test)

        ### Make Sparse Matrix
        # pt: playlist-tag sparse matrix
        # ps: playlist-song sparse matrix
        self.pt_train = to_sparse_matrix(df_train, self.playlist2idx, self.tag2idx, 'tags')
        self.ps_train = to_sparse_matrix(df_train, self.playlist2idx, self.song2idx, 'songs')

        self.pt_test = to_sparse_matrix(df_test, self.playlist2idx, self.tag2idx, 'tags', correction=self.n_train)
        self.ps_test = to_sparse_matrix(df_test, self.playlist2idx, self.song2idx, 'songs', correction=self.n_train)

        ### IDF Transformation
        self.transformer_tag, self.pt_idf_train = transform_idf(self.pt_train)
        self.transformer_song, self.ps_idf_train = transform_idf(self.ps_train)

        print('\n*********Train Data*********')
        print('Shape of Playlist-Tag Sparse Matrix: \t{}'.format(self.pt_train.shape))
        print('Shape of Playlist-Song Sparse Matrix: \t{}'.format(self.ps_train.shape))
        print('\n*********Test Data*********')
        print('Shape of Playlist-Tag Sparse Matrix: \t{}'.format(self.pt_test.shape))
        print('Shape of Playlist-Song Sparse Matrix: \t{}\n'.format(self.ps_test.shape))

    def _prepare_methods(self):
        """ Prepare methods for playlist continuations.

        Prepare methods used for playlist continuation. There are three main methods.
            1. Idf KNN Method
                - Find most similar playlsit based on idf transformed vector of tag and song.
                - consine similairy weigthed sum of k most similar playlist tag/song vectors
                - calculate playlist in test to playlist in train consine similarity based on idf transformed vector of tag and song.
            2. Item CF Method
                - Find item(tag/song) by item(tag/song) similarity based on item(tag/song) to playlist sparse matrix.
                - For every single seed items(tag/song) idf value weighted sum of similairty.
                - calculate item(tag/song) to item(tag/song) similarity based on idf transforemd vector of playlist.
            3. ALS Matrix Factorization Method
                - ALS Matrix Factorization Model
                - fit model on combined train and test sparse matrix.

        Additional Method for cold start problem.
            1. Title Method
                - If playlist has no seed item(tag/song) use title information to find similar playlist
        """        

        for method in self.methods:
            print('Preparing Method\t{}...'.format(method.name))        
            method.initialize(
                self.n_train, self.n_test, 
                self.pt_train, self.ps_train, self.pt_test, self.ps_test, 
                self.transformer_tag, self.transformer_song
            )
            # method.train()  # merge into initialize

        # Additional data structure necessary for title method
        print('Preparing Method\t{}...'.format(self.title_method.name))        
        self.title_method.playlist2idx = self.playlist2idx
        self.title_method.title2playlist = self.title2playlist
        self.title_method.initialize(
            self.n_train, self.n_test, 
            self.pt_train, self.ps_train, self.pt_test, self.ps_test, 
            self.transformer_tag, self.transformer_song
        )

    def _select_items(self, rating, top_item, already_item, idx2item, n):
        """ Select top items(tag/song) and conver to real name.

        Select top items(tag/song) based on ratings made by method.
        Convert index to real item(tag/song) name.

        Args:
            rating (ndarray)    : array of ratings on items(tag/song).
            top_item (ndarray)  : ordered items(tag/song) based on frequency in train data.
            already_item (ndarray)  : items(tag/song) in test data.
            idx2item (dict) : index to item name dictionary.
            n (int) : number of items to be selected. 
        Return:
            items (list)    : list of real name of items selected. 
        """ 

        idx = np.zeros(shape=(n))
        counter = 0
        for item_id in rating.argsort()[::-1]:
            if rating[item_id] == 0 or counter == n:
                break

            if item_id not in already_item:
                idx[counter] = item_id
                counter += 1

        ### fill with most popular items.                
        for item_id in top_item:
            if counter == n:
                break

            if item_id not in already_item and item_id not in idx:
                idx[counter] = item_id
                counter += 1

        items = [idx2item[item_id] for item_id in idx]
        return items

    def _generate_answers(self):
        """ Generate answers for playlist continuation task.

        Make rating results by methods in each playlist in test dataset. 
        Weighted combine the ratings afterward.
        Finally select 100 items for songs and 10 items for tags.

        Args:
        Return:
            answers (dict)  : selected items for playlist continuation.
        """        

        idx2tag = {idx:tag for tag, idx in self.tag2idx.items()}
        idx2song = {idx:song for song, idx in self.song2idx.items()}
        idx2playlist = {
            idx - self.n_train:playlist for playlist, idx in self.playlist2idx.items() if idx >= self.n_train
        }   # Only consider playlist in test dataset, which is to be recommended.

        answers = []

        ### Combine ratings by methods for continuation.
        for pid in tqdm(range(self.n_test)):

            rating_tag = np.zeros(shape=(len(self.tag2idx)))
            rating_song = np.zeros(shape=(len(self.song2idx)))

            for method, weight in zip(self.methods, self.weights):

                weight_tag = weight[0]
                weight_song = weight[1]

                rt, rs = method.predict(pid)

                rt = normalize_zero_to_one(rt)
                rs = normalize_zero_to_one(rs)

                rating_tag += (rt * weight_tag)
                rating_song += (rs * weight_song)

            ## Cold Start Problem. No seed tag in test.
            if len(self.pt_test[pid, :].nonzero()[1]) == 0:
                rt = self.title_method.predict(pid, 'tags')
                rt = normalize_zero_to_one(rt)

                ### more weights if there is no recommendation from other methods.
                if len(rating_tag.nonzero()[0]) == 0:
                    rating_tag += (rt * 0.8)
                else:
                    rating_tag += (rt * 0.2)

            ## Cold Start Problem. No seed song in test.
            if len(self.ps_test[pid, :].nonzero()[1]) == 0:
                rs = self.title_method.predict(pid, 'songs')
                rs = normalize_zero_to_one(rs)

                ### more weights if there is no recommendation from other methods.
                if len(rating_song.nonzero()[0]) == 0:
                    rating_song += (rs * 0.8)
                else:
                    rating_song += (rs * 0.2)

            ### get real playlist name
            playlist = idx2playlist[pid]

            ### get real tag name
            top_tag = self.transformer_tag.idf_.argsort()
            already_tag = self.pt_test[pid, :].nonzero()[1]
            tags = self._select_items(rating_tag, top_tag, already_tag, idx2tag, 10)

            ### get real song name
            top_song = self.transformer_song.idf_.argsort()
            already_song = self.ps_test[pid, :].nonzero()[1]
            songs = self._select_items(rating_song, top_song, already_song, idx2song, 100)

            ### make playlist continaution answers
            answers.append({
                "id": playlist,
                "songs": songs,
                "tags": tags,
            })

        return answers  

    def run(self, train_fname, test_fname):
        """ Maing method to be fired.

        Entry point for playlist continuation tast.

        Args:
            train_fname (str)   : train filename.
            test_fname  (str)   : test filename.
        Return:
        """

        print("Loading train file...")
        train = load_json(train_fname)

        print("Loading test file...")
        test = load_json(test_fname)

        print('Preparing data...')
        self._prepare_data(train, test)

        print('Preparing methods...')
        als_params = {
            'tag': {
                'factors': 512,      
                'regularization': 0.001,
                'iterations': 150, 
                'confidence': 100
            },
            'song': {
                'factors': 512, 
                'regularization': 0.001,
                'iterations': 50, 
                'confidence': 100
            }
        }

        self.methods = [
            ItemCFMethod('item-collaborative-filtering'),   
            IdfKNNMethod('idf-knn', k_ratio=0.001),        
            ALSMFMethod('als-matrix-factorization', params=als_params),
        ]
        self.weights = [
            (0.22, 0.30),
            (0.4, 0.36),
            (0.38, 0.34)
        ]
        self._prepare_methods()

        print('Generating answers...')
        answers = self._generate_answers()

        print("Writing answers...")
        write_json(answers, "results/results.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('train_fname', type=str, metavar='Train File Name', default='./res/train.json')
    parser.add_argument('test_fname', type=str, metavar='Test File Name', default='./res/test.json')
    args = parser.parse_args()

    train_fname = args.train_fname
    test_fname = args.test_fname

    inference = PlaylistContinuation()
    inference.run(train_fname=train_fname, test_fname=test_fname)

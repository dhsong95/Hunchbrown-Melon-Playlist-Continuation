# -*- coding: utf-8 -*-
from collections import Counter

import fire
from tqdm import tqdm

import pandas as pd

from arena_util import load_json
from arena_util import write_json
from arena_util import remove_seen
from arena_util import most_popular

from model_util import json_to_dataframe
from model_util import get_unique_items
from model_util import make_item_index_dictionary
from model_util import dataframe_to_matrix
from model_util import transform_sparse_matrix_tfidf
from model_util import find_neigbors
from model_util import recommend_items


class ModelKNNTFIDF:
    def _generate_answers(self, train, questions):
        print("Preprocessing data... > to dataframe")
        train = json_to_dataframe(train)
        test = json_to_dataframe(questions)

        N_TRAIN = len(train)
        N_TEST = len(test)

        print("Preprocessing data... > to sparse matrix")
        unique_tags = get_unique_items(pd.concat([train, test], ignore_index=True, copy=False), 'tags', list_type=True)
        unique_songs = get_unique_items(pd.concat([train, test], ignore_index=True, copy=False), 'songs', list_type=True)
        unique_playlists_train = get_unique_items(train, 'id', list_type=False)
        unique_playlists_test = get_unique_items(test, 'id', list_type=False)

        tag2idx, idx2tag = make_item_index_dictionary(unique_tags)
        song2idx, idx2song = make_item_index_dictionary(unique_songs)

        playlist2idx_train, idx2playlist_train = make_item_index_dictionary(unique_playlists_train)
        playlist2idx_test, idx2playlist_test = make_item_index_dictionary(unique_playlists_test)

        playlist2idx = {playlist:idx for playlist, idx in playlist2idx_train.items()}
        for playlist, idx in playlist2idx_test.items():
            playlist2idx[playlist] = (idx + N_TRAIN)
        idx2playlist = {idx:playlist for playlist, idx in playlist2idx.items()}

        assert len(playlist2idx_train) + len(playlist2idx_test) == len(playlist2idx)

        PT_train = dataframe_to_matrix(train, item='tags', playlist2idx=playlist2idx_train, item2idx=tag2idx)
        PT_test = dataframe_to_matrix(test, item='tags', playlist2idx=playlist2idx_test, item2idx=tag2idx)
        PT = dataframe_to_matrix(pd.concat([train, test], ignore_index=True, copy=False), item='tags', playlist2idx=playlist2idx, item2idx=tag2idx)

        PS_train = dataframe_to_matrix(train, item='songs', playlist2idx=playlist2idx_train, item2idx=song2idx)
        PS_test = dataframe_to_matrix(test, item='songs', playlist2idx=playlist2idx_test, item2idx=song2idx)
        PS = dataframe_to_matrix(pd.concat([train, test], ignore_index=True, copy=False), item='songs', playlist2idx=playlist2idx, item2idx=song2idx)

        _, PT_tfidf_train, PT_tfidf_test = transform_sparse_matrix_tfidf(PT, PT_train, PT_test)
        _, PS_tfidf_train, PS_tfidf_test = transform_sparse_matrix_tfidf(PS, PS_train, PS_test)
        
        print("Prepare for recommendations... > find neighbors")
        _, neighbors = find_neigbors(PT_tfidf_train, PT_tfidf_test, PS_tfidf_train, PS_tfidf_test, k=10)

        print("Writing answers...")
        answers = list()
        for rid in tqdm(range(N_TEST)):
            playlist = idx2playlist_test[rid]
            rid_neighbors = neighbors[rid, :]

            tags = recommend_items(rid, PT_tfidf_train, PT_tfidf_test, rid_neighbors, idx2tag, 10)
            songs = recommend_items(rid, PS_tfidf_train, PS_tfidf_test, rid_neighbors, idx2song, 100)

            answers.append({
                "id": playlist,
                "songs": songs,
                "tags": tags,
            })

        return answers

    def run(self, train_fname, question_fname):
        print("Loading train file...")
        train_data = load_json(train_fname)

        print("Loading question file...")
        test_data = load_json(question_fname)

        answers = self._generate_answers(train_data, test_data)
        write_json(answers, "results/results.json")


if __name__ == "__main__":
    fire.Fire(ModelKNNTFIDF)

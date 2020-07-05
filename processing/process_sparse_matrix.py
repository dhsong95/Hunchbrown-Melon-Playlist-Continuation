# -*- coding: utf-8 -*-
"""
Author: DH Song
Last Modified: 2020.06.25
"""

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer


def to_sparse_matrix(dataframe, playlist2idx, item2idx, mode, correction=0):
    '''
        change dataframe to user item sparse matrix

    Args:
        dataframe(pandas DataFrame): dataframe to be converted
        playlist2idx(dict): playlist-playlist index dictionary
        item2idx(dict): tag-tag index or song-song index dictionary
        mode(str): tags or songs to be item part
        correction(int): train and test is vertically stacked in playlist2idx. 
                         For test playlsit to allocate right index. need adjustment
    Return:
        sparse_matrix(scipy csr matrix): user item sparse matrix 
    '''
    assert mode in ['tags', 'songs']

    rows = list()
    cols = list()
    data = list()

    for idx in range(len(dataframe)):
        pid = playlist2idx[dataframe.loc[idx, 'id']] - correction
        assert pid == idx

        for item in dataframe.loc[idx, mode]:
            rows.append(pid)
            cols.append(item2idx[item])
        
    assert len(rows) == len(cols)
    rows = np.array(rows)
    cols = np.array(cols)
    data = np.ones(len(rows))

    sparse_matrix = sp.csr_matrix((data, (rows, cols)), shape=(len(dataframe), len(item2idx)))
        
    return sparse_matrix


def transform_idf(sparse_matrix):
    '''
        idf transformation of sparse matrix

    Args:
        sparse_matrix(scipy csr matrix): user item sparse matrix 
    Return:
        transformer(sci-kit learn TfIdfTransformer): TfIdfTransformer model
        sparse_matrix_idf(scipy csr matrix): TfIdf converted user item sparse matrix 
    '''
    transformer = TfidfTransformer(smooth_idf=True)
    transformer.fit(sparse_matrix)
    sparse_matrix_idf = transformer.transform(sparse_matrix)

    return transformer, sparse_matrix_idf


def horizontal_stack(left, right, weights):
    '''
        horizontally stacking with weight

    Args:
        left(scipy csr matrix): user item sparse matrix (tags)
        right(scipy csr matrix): user item sparse matrix (songs)
        weights(list): weight for left and right
    Return:
        weighted_stacked(scipy csr matrix): Weighted Stacked user item sparse matrix 
    '''
    weighted_stacked = sp.hstack([left * weights[0], right * weights[1]])
    return weighted_stacked

def vertical_stack(up, down):
    '''
        vertically stacking

    Args:
        up(scipy csr matrix): user item sparse matrix (train)
        down(scipy csr matrix): user item sparse matrix (test)
    Return:
        stacked(scipy csr matrix): Stacked user item sparse matrix 
    '''
    stacked = sp.vstack([up, down])
    return stacked

def write_sparse_matrix(data, fname):
    '''
        save sparse matrix

    Args:
        data(scipy csr matrix): sparse matrix to be saved. usually similarity
        fname(str): local filename 
    Return:
    '''
    sp.save_npz(fname, data)


def load_sparse_matrix(fname):
    '''
        load sparse matrix

    Args:
        fname(str): local filename 
    Return:
        data(scipy csr matrix): sparse matrix to be loaded. usually similarity
    '''
    data = sp.load_npz(fname)
    return data
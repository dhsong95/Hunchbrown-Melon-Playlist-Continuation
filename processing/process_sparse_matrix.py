# -*- coding: utf-8 -*-
""" About processing Sparse Matrix 

Author: Hunchbrown - DH Song
Last Modified: 2020.07.20

About processing Sparse Matrix
"""

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer

def to_sparse_matrix(dataframe, playlist2idx, item2idx, mode, correction=0):
    """ to sparse matrix(csr_matrix)

    convert DataFrame to csr_matrix

    Args:
        dataframe (DataFrame)   : dataframe to be converted
        playlist2idx (dict) : playlist-playlist index dictionary
        item2idx (dict) : tag-tag index or song-song index dictionary
        mode (str)  : tags or songs to be item part
        correction (int)    : train and test is vertically stacked in playlist2idx, therefore to allocate right index need adjustment.
    Return:
        sparse_matrix (csr_matrix)  : user item sparse matrix 
    """

    assert mode in ['tags', 'songs']

    rows = list()
    cols = list()
    data = list()

    for idx in range(len(dataframe)):
        pid = playlist2idx[dataframe.loc[idx, 'id']] - correction
        assert pid == idx

        for item in dataframe.loc[idx, mode]:
            if item in item2idx:
                rows.append(pid)
                cols.append(item2idx[item])
        
    assert len(rows) == len(cols)

    rows = np.array(rows)
    cols = np.array(cols)
    data = np.ones(len(rows))

    sparse_matrix = sp.csr_matrix((data, (rows, cols)), shape=(len(dataframe), len(item2idx)))
        
    return sparse_matrix


def transform_idf(sparse_matrix):
    """ TfIdf transformation

    tf-idf transform sparse matrix

    Args:
        sparse_matrix (csr_matrix)  : user item sparse matrix 
    Return:
        transformer (TfIdfTransformer)  : TfIdfTransformer model
        sparse_matrix_idf (csr_matrix)  : TfIdf converted user item sparse matrix 
    """

    transformer = TfidfTransformer(smooth_idf=True)
    transformer.fit(sparse_matrix)
    sparse_matrix_idf = transformer.transform(sparse_matrix)

    return transformer, sparse_matrix_idf

def horizontal_stack(left, right, weights):
    """ Horizontal Stack

    horizontally stacking with weights

    Args:
        left (csr_matrix)   : user item sparse matrix (tags)
        right (csr_matrix)  : user item sparse matrix (songs)
        weights (list)  : weight for left and right
    Return:
        weighted_stacked(csr_matrix): Weighted Stacked user item sparse matrix 
    """

    weighted_stacked = sp.hstack([left * weights[0], right * weights[1]])
    return weighted_stacked

def vertical_stack(up, down):
    """ Vertical Stack

    vertically stacking

    Args:
        up (csr_matrix) : user item sparse matrix (train)
        down (csr_matrix)   : user item sparse matrix (test)
    Return:
        stacked (csr_matrix)    : Stacked user item sparse matrix 
    """
    stacked = sp.vstack([up, down])
    return stacked

def write_sparse_matrix(data, fname):
    """ Save Sparse Matrix

    save sparse matrix in npz forms.

    Args:
        data (csr_matrix): sparse matrix to be saved. usually similarity
        fname (str) : local filename 
    Return:
    """
    sp.save_npz(fname, data)


def load_sparse_matrix(fname):
    """ Load Sparse Matrix

    load sparse matrix in npz forms.

    Args:
        fname (str) : local filename 
    Return:
        data (csr_matrix)   : sparse matrix to be loaded. usually similarity
    """

    data = sp.load_npz(fname)
    return data
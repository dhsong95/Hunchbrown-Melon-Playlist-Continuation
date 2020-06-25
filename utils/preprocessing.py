# -*- coding: utf-8 -*-


from tqdm import tqdm
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfTransformer




def to_sparse_matrix(dataframe, playlist2idx, item2idx, name, correction=0):
    assert name in ['tags', 'songs']

    rows = list()
    cols = list()
    data = list()

    for idx in tqdm(range(len(dataframe))):
        pid = playlist2idx[dataframe.loc[idx, 'id']] - correction
        assert pid == idx

        for item in dataframe.loc[idx, name]:
            rows.append(pid)
            cols.append(item2idx[item])
        
    assert len(rows) == len(cols)
    rows = np.array(rows)
    cols = np.array(cols)
    data = np.ones(len(rows))

    sparse_matrix = sp.csr_matrix((data, (rows, cols)), shape=(len(dataframe), len(item2idx)))
        
    return sparse_matrix

def transform_idf(sparse_matrix, transform=True):
        transformer = TfidfTransformer(smooth_idf=True)
        transformer.fit(sparse_matrix)
        if transform:
            sparse_matrix = transformer.transform(sparse_matrix)

        return transformer, sparse_matrix

from tqdm import tqdm

import pandas as pd
import numpy as np
import scipy.sparse as sp

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity

def json_to_dataframe(json_data):
    dataframe_dict = {'id': [], 'plylst_title': [], 'tags': [], 'songs': [], 'like_cnt': [], 'updt_date': []}

    for data in tqdm(json_data):
        dataframe_dict['id'].append(data['id'])
        dataframe_dict['plylst_title'].append(data['plylst_title'])
        dataframe_dict['tags'].append(data['tags'])
        dataframe_dict['songs'].append(data['songs'])
        dataframe_dict['like_cnt'].append(data['like_cnt'])
        dataframe_dict['updt_date'].append(data['updt_date'])
    
    dataframe = pd.DataFrame(dataframe_dict)
    dataframe['updt_date'] = pd.to_datetime(dataframe.updt_date)

    return dataframe


def get_unique_items(dataframe, column, list_type=True):
    unique_items = set()
    if list_type:
        for c in tqdm(dataframe[column]):
            unique_items |= set(c)
    else:
        assert len(dataframe[column].unique()) == len(dataframe[column])
        unique_items = dataframe[column].unique()
    
    return unique_items


def make_item_index_dictionary(items):
    item2idx = {item:idx for idx, item in enumerate(items)}
    idx2item = {idx:item for item, idx in item2idx.items()}
    return item2idx, idx2item


def dataframe_to_matrix(dataframe, item='tags', playlist2idx=None, item2idx=None):
    assert item in ['tags', 'songs']

    matrix_shape = (len(playlist2idx), len(item2idx))

    if 'plylst_id' not in dataframe.columns:
        dataframe['plylst_id'] = dataframe.id.map(playlist2idx)

    column_name = '{}_id'.format(item)
    if column_name not in dataframe.columns:
        dataframe[column_name] = dataframe[item].apply(lambda items: [item2idx[item] for item in items])

    rows = list()
    cols = list()
    data = list()

    for r, cs in tqdm(zip(dataframe.plylst_id, dataframe[column_name])):
        for c in cs:
            rows.append(r)
            cols.append(c)
    
    rows = np.array(rows)
    cols = np.array(cols)
    data = np.ones(rows.shape[0])

    return sp.csr_matrix((data, (rows, cols)), shape=matrix_shape)

def transform_sparse_matrix_tfidf(train_test, train, test):
    transformer = TfidfTransformer(smooth_idf=True)
    transformer.fit(train_test)
    tfidf_train = transformer.transform(train)
    tfidf_test = transformer.transform(test)

    return transformer, tfidf_train, tfidf_test


def _calculate_cosine_similarity(A, B):
    return cosine_similarity(A, B, dense_output=False)

### fint neighbors of test from train
def find_neigbors(tag_train, tag_test, song_train, song_test, k=5):
    train = sp.hstack([tag_train, song_train])
    test = sp.hstack([tag_test, song_test])

    similarity = _calculate_cosine_similarity(test, train)
    neighbors = list()
    for rid in tqdm(range(similarity.shape[0])):
        neighbors.append(np.argsort(similarity[rid, :].toarray()[0])[::-1][:k])
    
    neighbors = np.array(neighbors)
    return similarity, neighbors

def recommend_items(id, train, test, neighbors, idx2item, n):
    counter = 0
    items = list()
    for item_id in train[neighbors, :].toarray().sum(axis=0).argsort()[::-1]:
        if item_id not in test[id, :].nonzero()[1]:
            item = idx2item[item_id]
            items.append(item)
            counter += 1
        if counter == n:
            break
    return items

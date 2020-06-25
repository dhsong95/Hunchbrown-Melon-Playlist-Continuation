# -*- coding: utf-8 -*-
"""
Author: DH Song
Last Modified: 2020.06.25
"""

import pandas as pd

def _get_unique_items(dataframe, name, list_type=True):
    '''
        get unique recrod in dataframe column

    Args:
        dataframe(pandas DataFrame): dataframe to be checked
        name(str): column name
        list_type(bool): if list_type is true. need to unpack the item in list
    Return:
        unique_items(list): list of unique items(tags or songs) 
    '''
    unique_items = set()
    if list_type:
        for item in dataframe[name]:
            unique_items |= set(item)
        # sorting is done for tag ans song
        unique_items = sorted(unique_items)
    else:
        # assertion for playlist: guarantees that playlist has unique id in dataframe
        assert len(dataframe[name].unique()) == len(dataframe[name])
        # vertically stacked
        unique_items = dataframe[name]
    
    return unique_items


def to_dataframe(data):
    '''
        change json to dataframe

    Args:
        data(json data type): source of json file
    Return:
        dataframe(pandas DataFrame): destination of json file in forms of pandas DataFrame 
    '''
    items = {'id': [], 'plylst_title': [], 'tags': [], 'songs': [], 'like_cnt': [], 'updt_date': []}

    for item in data:
        items['id'].append(item['id'])
        items['plylst_title'].append(item['plylst_title'])
        items['tags'].append(item['tags'])
        items['songs'].append(item['songs'])
        items['like_cnt'].append(item['like_cnt'])
        items['updt_date'].append(item['updt_date'])
    
    dataframe = pd.DataFrame(items)
    dataframe['updt_date'] = pd.to_datetime(dataframe.updt_date)

    return dataframe


def get_item_idx_dictionary(train, test, mode):
    '''
        give unique index(start from 0) for item.

    Args:
        train(pandas DataFrame): train dataframe
        test(pandas DataFrame): test dataframe
        mode(str): mode determines which item to be converted. tags, songs, id possible
    Return:
        item2idx(dict): item-index dictionary 
    '''
    assert mode in ['tags', 'songs', 'id']
    items = _get_unique_items(pd.concat([train, test], ignore_index=True), mode, mode in ['tags', 'songs'])
    item2idx = {item:idx for idx, item in enumerate(items)}
    return item2idx

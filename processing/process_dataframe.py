# -*- coding: utf-8 -*-
""" About processing DataFrame 

Author: Hunchbrown - DH Song
Last Modified: 2020.07.20

About processing DataFrame
"""

import pandas as pd

def _get_unique_items(dataframe, name, list_type=True):
    """ return unique items.

    return unique items in dataframe.

    Args:
        dataframe (DataFrame)   : dataframe to be checked
        name (str)  : column name
        list_type (bool)    : if list_type is true. need to unpack the item in list
    Return:
        unique_items (list) : list of unique items(tags/songs) 
    """

    unique_items = set()
    if list_type:
        for item in dataframe[name]:
            unique_items |= set(item)

        # sorting is done for tag and song
        unique_items = sorted(unique_items)
    else:
        # assertion for playlist: guarantees that playlist has unique id in dataframe
        assert len(dataframe[name].unique()) == len(dataframe[name])

        # vertically aligned(train -> test)
        unique_items = dataframe[name]
    
    return unique_items


def to_dataframe(data):
    """ to dataframe

    convert json to dataframe

    Args:
        data (json) : source of json file
    Return:
        dataframe (DataFrame)   : destination of json file in forms of pandas DataFrame 
    """

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


def get_item_idx_dictionary(train, test=None, mode='tags'):
    """ Return dictionary

    return item to index dictionary

    Args:
        train (DataFrame)   : train dataframe
        test (DataFrame)    : test dataframe
        mode (str)  : mode determines which item to be converted. tags, songs, id possible
    Return:
        item2idx(dict): item to index dictionary 
    """

    assert mode in ['tags', 'songs', 'id']
    if test is None:
        items = _get_unique_items(train, mode, mode in ['tags', 'songs'])
    else:
        items = _get_unique_items(pd.concat([train, test], ignore_index=True), mode, mode in ['tags', 'songs'])
    item2idx = {item:idx for idx, item in enumerate(items)}
    return item2idx

def map_title_to_playlist(train, test):
    """ Return dictionary

    return title to list of playlist dictionary

    Args:
        train (DataFrame)   : train dataframe
        test (DataFrame)    : test dataframe
    Return:
        title2playlist(dict)    : title to list of plalist dictionary 
    """

    title2playlist = dict()
    df = pd.concat([train, test], ignore_index=True)
    for idx in range(len(df)):
        title = df.loc[idx, 'plylst_title']
        playlist = df.loc[idx, 'id']
        if title not in title2playlist:
            title2playlist[title] = list()
        title2playlist[title].append(playlist)

    return title2playlist
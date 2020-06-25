# -*- coding: utf-8 -*-

import pandas as pd


def _get_unique_items(dataframe, name, list_type=True):
    unique_items = set()
    if list_type:
        for item in dataframe[name]:
            unique_items |= set(item)
        unique_items = sorted(unique_items)
    else:
        # assertion for playlist: guarantees that playlist has unique id in dataframe
        assert len(dataframe[name].unique()) == len(dataframe[name])
        unique_items = dataframe[name]
    
    return unique_items


def to_dataframe(data):
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


def get_item_idx_dictionary(train, test, name):
    assert name in ['tags', 'songs', 'id']
    items = _get_unique_items(pd.concat([train, test], ignore_index=True), name, name in ['tags', 'songs'])
    item2idx = {item:idx for idx, item in enumerate(items)}
    return item2idx

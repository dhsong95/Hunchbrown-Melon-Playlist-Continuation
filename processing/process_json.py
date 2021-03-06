# -*- coding: utf-8 -*-
""" About processing json 

Author: Hunchbrown - DH Song based on Kakao Provided
Last Modified: 2020.07.20

About processing DataFrame
"""

import io
import os
import json
import distutils.dir_util

import numpy as np


def write_json(data, fname):
    """ save json file

    convert variable to json

    Args:
        data (str)  : data to be saved
        fname (str) : local filename to be saved
    Return:
    """

    def _conv(o):
        if isinstance(o, (np.int64, np.int32)):
            return int(o)
        raise TypeError

    parent = os.path.dirname(fname)
    distutils.dir_util.mkpath("./arena_data/" + parent)
    with io.open("./arena_data/" + fname, "w", encoding="utf-8") as f:
        json_str = json.dumps(data, ensure_ascii=False, default=_conv)
        f.write(json_str)


def load_json(fname):
    """ load json file

    load data from json file.

    Args:
        fname (str): local json filename to be loaded

    Return:
        jsong_obj (dict): json file
    """

    with open(fname, encoding="utf-8") as f:
        json_obj = json.load(f)

    return json_obj


def debug_json(r):
    """ json debugging

    check json before writing

    Args:
        r (str): local json filename to be checked
    Return:
    """

    print(json.dumps(r, ensure_ascii=False, indent=4))

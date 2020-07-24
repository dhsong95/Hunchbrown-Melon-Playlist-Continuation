# -*- coding: utf-8 -*-
""" Normalization. 

Author: Hunchbrown - DH Song
Last Modified: 2020.07.17

Normalizing Method.
"""

def normalize_zero_to_one(rating):
    """ Normalize zero to one. 

    normalize zero to one based on min/max value

    Args:
        rating (ndarray)    : rating array to be normalized
    return:
        rating (ndarray)    : normalized rating array
    """    

    rating_min = rating.min(-1)
    rating_max = rating.max(-1)
    if rating_max != 0:
        rating = (rating - rating_min) / (rating_max - rating_min)
    return rating
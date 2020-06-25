# -*- coding: utf-8 -*-
"""
Author: DH Song
Last Modified: 2020.06.25
"""

from sklearn.metrics.pairwise import cosine_similarity

def calculate_cosine_similarity(*args):
    '''
        calculate cosine similarity of sparse matrix

    Args:
        *args: sparse matrix
    Return:
        : calculated cosine similarity. not in dense output for memory 
    '''
    if len(args) == 1:
        return cosine_similarity(args[0], dense_output=False)
    else:
        return cosine_similarity(args[0], args[1], dense_output=False)
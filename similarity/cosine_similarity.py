# -*- coding: utf-8 -*-
""" Similarity Method. 

Author: Hunchbrown - DH Song
Last Modified: 2020.07.17

Calculate Similarity.
"""

from sklearn.metrics.pairwise import cosine_similarity

def calculate_cosine_similarity(*args):
    """ Cosine Similarity

    calculate cosine similarity of sparse matrix

    Args:
        *args: sparse matrix
    Return:
        similarity : calculated cosine similarity. not in dense output for memory 
    """

    if len(args) == 1:
        similarity = cosine_similarity(args[0], dense_output=False)
    else:
        similarity = cosine_similarity(args[0], args[1], dense_output=False)
    
    return similarity
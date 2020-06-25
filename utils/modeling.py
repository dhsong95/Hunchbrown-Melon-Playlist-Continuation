# -*- coding: utf-8 -*-


import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp


def calculate_cosine_similarity(*args):
    if len(args) == 1:
        return cosine_similarity(args[0], dense_output=False)
    else:
        return cosine_similarity(args[0], args[1], dense_output=False)


def horizontal_stack_sparse_matrix(tag, song):
    return sp.hstack([tag * 0.15, song * 0.85])
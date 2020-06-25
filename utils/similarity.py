# -*- coding: utf-8 -*-

from sklearn.metrics.pairwise import cosine_similarity

def calculate_cosine_similarity(*args):
    if len(args) == 1:
        return cosine_similarity(args[0], dense_output=False)
    else:
        return cosine_similarity(args[0], args[1], dense_output=False)

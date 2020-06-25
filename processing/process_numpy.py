# -*- coding: utf-8 -*-

import numpy as np

def get_zero_array(shape):
    return np.zeros(shape=shape)

def save_numpy(filename, array):
    np.save(filename, array)

def load_numpy(filename, weight):
    arr = np.load(file=filename)
    return arr * weight
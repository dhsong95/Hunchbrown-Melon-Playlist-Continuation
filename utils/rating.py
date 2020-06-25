import numpy as np


def item_cf_rating(pid, test, similarity, idf):
    rating = np.zeros(shape=(test.shape[1]))
    for item in test[pid, :].nonzero()[1]:
        rating += (similarity[item, :] * idf[item]).toarray().reshape(-1)
    return rating


def idf_knn_rating(pid, train, test, neighbors, similarity):
    rating = np.zeros(test.shape[1])
    for neighbor in neighbors[pid, :]:
        s = similarity[pid, neighbor]
        rating += (s * train[neighbor, :]).toarray().reshape(-1)  
    
    return rating

def mf_rating(pid, test, model, idf):
    rating = np.zeros(test.shape[1])
    item_features = model.item_factors
    user_feature = np.zeros(item_features.shape[1])

    if test[pid, :].count_nonzero() != 0:
        denominator = 0.0
        for item in test[pid, :].nonzero()[1]:
            user_feature += (idf[item] * item_features[item])
            denominator += idf[item]
        user_feature /= denominator
    
        rating = np.dot(user_feature, item_features.T)
    return rating

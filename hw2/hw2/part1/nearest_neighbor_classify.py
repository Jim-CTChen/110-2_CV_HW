from __future__ import print_function

import numpy as np
import scipy.spatial.distance as distance
from scipy.stats import mode

def nearest_neighbor_classify(train_image_feats, train_labels, test_image_feats):
    ###########################################################################
    # TODO:                                                                   #
    # This function will predict the category for every test image by finding #
    # the training image with most similar features. Instead of 1 nearest     #
    # neighbor, you can vote based on k nearest neighbors which will increase #
    # performance (although you need to pick a reasonable value for k).       #
    ###########################################################################
    ###########################################################################
    # NOTE: Some useful functions                                             #
    # distance.cdist :                                                        #
    #   This function will calculate the distance between two list of features#
    #       e.g. distance.cdist(? ?)                                          #
    ###########################################################################
    '''
    Input : 
        train_image_feats : 
            image_feats is an (N, d) matrix, where d is the 
            dimensionality of the feature representation.

        train_labels : 
            image_feats is a list of string, each string
            indicate the ground truth category for each training image. 

        test_image_feats : 
            image_feats is an (M, d) matrix, where d is the 
            dimensionality of the feature representation.
    Output :
        test_predicts : 
            a list(M) of string, each string indicate the predict
            category for each testing image.
    '''
    print('Start classifying...')

    K = 5 # for kNN
    test_predicts = []
    for test_feat in test_image_feats:
        distance_list = np.array([distance.cdist(np.expand_dims(test_feat, axis=0), np.expand_dims(train_feat, axis=0), 'minkowski', p=0.6) for train_feat in train_image_feats]).flatten()
        top_k = distance_list.argsort()[:K] # get shortest k distance
        category = {}
        for idx in top_k:
            try:
                category[train_labels[idx]] += 1
            except KeyError:
                category[train_labels[idx]] = 1
        max = 0
        predict = ''
        for k, v in category.items():
            if v > max: 
                max = v
                predict = k
        test_predicts.append(predict)
    #############################################################################
    #                                END OF YOUR CODE                           #
    #############################################################################
    return test_predicts

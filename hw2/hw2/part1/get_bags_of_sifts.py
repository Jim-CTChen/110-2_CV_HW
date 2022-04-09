from PIL import Image
import numpy as np
import pickle
import scipy.spatial.distance as distance
from cyvlfeat.sift.dsift import dsift
from time import time

def get_bags_of_sifts(image_paths):
    ############################################################################
    # TODO:                                                                    #
    # This function assumes that 'vocab.pkl' exists and contains an N x 128    #
    # matrix 'vocab' where each row is a kmeans centroid or visual word. This  #
    # matrix is saved to disk rather than passed in a parameter to avoid       #
    # recomputing the vocabulary every time at significant expense.            #
    #                                                                          #                                                               
    # image_feats is an N x d matrix, where d is the dimensionality of the     #
    # feature representation. In this case, d will equal the number of clusters#
    # or equivalently the number of entries in each image's histogram.         #
    #                                                                          #
    # You will construct SIFT features here in the same way you did in         #
    # build_vocabulary (except for possibly changing the sampling rate)        #
    # and then assign each local feature to its nearest cluster center         #
    # and build a histogram indicating how many times each cluster was used.   #
    # Don't forget to normalize the histogram, or else a larger image with more#
    # SIFT features will look very different from a smaller version of the same#
    # image.                                                                   #
    ############################################################################
    '''
    Input : 
        image_paths : a list(N) of training images
    Output : 
        image_feats : (N, d) feature, each row represent a feature of an image
    '''
    print('Start extracting feature in training set...')

    image_feats = []

    with open('vocab.pkl', 'rb') as handle:
        vocab = pickle.load(handle)

    num_of_cluster = vocab.shape[0]

    for path in image_paths:
        img = np.array(Image.open(path))
        _, descriptors = dsift(img, step=5, fast=True)
        nearest_cluster = []
        for descriptor in descriptors:
            # find nearest cluster for each feature(descriptor)
            distance_list = np.array([distance.cdist(np.expand_dims(descriptor, axis=0), np.expand_dims(v, axis=0), 'euclidean') for v in vocab]).flatten()
            nearest_cluster.append(distance_list.argmin())
        cluster_histogram, _ = np.histogram(nearest_cluster, bins=np.arange(num_of_cluster))
        image_feats.append(cluster_histogram.tolist())

    
    #############################################################################
    #                                END OF YOUR CODE                           #
    #############################################################################
    return image_feats

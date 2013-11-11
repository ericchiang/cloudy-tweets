#!/usr/bin/python

__author__    = "Eric Chiang"
__copyright__ = "Copyright 2013, Eric Chiang"
__email__     = "eric.chiang.m@gmail.com"

__license__   = "GPL"
__version__   = "3.0"

import numpy
import nltk
from scipy.sparse import lil_matrix, vstack

"""
Feature generator. Creates a sparce feature matrix from bag of words.

Each feature represents the number of times a word apears in a bag. The number
of features are limited to the top n words.
"""
class TopWordsFeatures(object):

    """
    Generate word counts and determine the top n words. This will fix the
    definition of the feature space.
    """
    def fit(self,X_bag_of_words):
        all_words = []
        for bag_of_words in X_bag_of_words:
            all_words.extend(bag_of_words)
        top_words = nltk.FreqDist(all_words).keys()[:self.m]
        print top_words
        for i in range(len(top_words)):
            self.word_indices[top_words[i]] = i
        return self

    """
    Generate a sparce matrix representing the feature space of a set of bag
    of words.
    """
    def generateFeatures(self,X_bag_of_words):
        features = []
        sparse_features = None
        # Limits the use of dense matrix before merging into sparce matrix 
        combine_limit = 10000
        for bag_of_words in X_bag_of_words:
            X_feats = [0.] * self.m
            for word in set(bag_of_words):
                try:
                    X_feats[self.word_indices[word]] += 1.
                except KeyError:
                    None
            features.append(X_feats)
            if len(features) >= combine_limit:
                if sparse_features is not None:
                    sparse_features = vstack([sparse_features,
                                              lil_matrix(features)])
                else:
                    sparse_features = lil_matrix(features)
                features = []
        if sparse_features is not None:
            sparse_features = vstack([sparse_features,lil_matrix(features)])
        else:
            sparse_features = lil_matrix(features)
            features = []

        return lil_matrix(sparse_features)


    def __init__(self,m=100):
        self.m = m
        self.word_indices = {}

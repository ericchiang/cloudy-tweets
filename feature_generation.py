import numpy
import nltk
from scipy.sparse import lil_matrix, vstack

"""
Class for generating feature matrices from bag-of-words.
"""
class BagOfWordsFeatures(object):

    """
    Generate training features. Will also 'fit' this class so further feature
    matrices of an identical format can be created.
    """
    def genTrainFeats(self,X_bag_of_words):
        if self.n != 0:
            raise Exception("Feature generator already trained")
        # Generate word counts
        word_counts = {}
        for bag_of_words in X_bag_of_words:
            for word in bag_of_words:
                try:
                    word_counts[word] += 1
                except KeyError:
                    word_counts[word] = 1
        for word in word_counts.keys():
            if word_counts[word] >= self.rare:
                self.word_indexes[word] = len(self.word_indexes)
        self.n = len(self.word_indexes)
        return self.genTestFeats(X_bag_of_words)

    """
    Generate test features using the same format created by the training
    features.
    """
    def genTestFeats(self,X_bag_of_words):
        features = None
        combine_limit = 10000
        for bag_of_words in X_bag_of_words:
            X_feats = [0.] * self.n
            for word in bag_of_words:
                try:
                    X_feats[self.word_indexes[word]] += 1.
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

    def __init__(self,rare=3):
        # min times a word must appear to be included in feature matrix
        self.rare = rare
        # index of word for feature matrix
        self.word_indexes = {}
        # number of features
        self.n = 0

class TopWordsFeatures(object):
    def fit(self,X_bag_of_words):
        all_words = []
        for bag_of_words in X_bag_of_words:
            all_words.extend(bag_of_words)
        top_words = nltk.FreqDist(all_words).keys()[:self.n]
        print top_words
        for i in range(len(top_words)):
            self.word_indexes[top_words[i]] = i
        return self

    def generateFeatures(self,X_bag_of_words):
        features = []
        sparse_features = None
        combine_limit = 10000
        for bag_of_words in X_bag_of_words:
            X_feats = [0.] * self.n
            for word in bag_of_words:
                try:
                    X_feats[self.word_indexes[word]] += 1.
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


    def __init__(self,n=100):
        self.n = n
        self.word_indexes = {}

#!/usr/bin/python

__author__    = "Eric Chiang"
__copyright__ = "Copyright 2013, Eric Chiang"
__email__     = "eric.chiang.m@gmail.com"

__license__   = "GPL"
__version__   = "3.0"

"""
File runs the cloudy-tweets pipeline, from feature generation to predicting
scores.
"""

import sys
import numpy as np
import pickle as pk
from sklearn.cross_validation import KFold
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn import svm
from feature_generation import TopWordsFeatures
from format_data import *
from utils import *


def usage():
    print """
        Usage:
        python %s [train.csv] [test.csv] [output]
    """

if __name__ == '__main__':
    if len(sys.argv) != 4:
        usage()
        sys.exit(2)

    # Check file validity
    try:
        train_csv = open(sys.argv[1], 'rb')
    except IOError:
        sys.stderr.write("[Error] Could not open file '%s'" % (sys.argv[1],))
        sys.exit(2)

    try:
        test_csv = open(sys.argv[2], 'rb')
    except IOError:
        sys.stderr.write("[Error] Could not open file '%s'" % (sys.argv[2],))
        sys.exit(2)

    num_feats = 4000

    # variables
    var_names = ['s1','s2','s3','s4','s5',
                 'w1','w2','w3','w4',
                 'k1','k2','k3','k4','k5','k6','k7','k8',
                 'k9','k10','k11','k12','k13','k14','k15']

    printInfo("Parsing training data")

    # Parse training data
    train_data = parseTwitterCSV(train_csv)
    train_header = train_data[0]
    train_data = train_data[1:]

    # Create train bag of words
    train_bag_of_words = []
    for row in train_data:
        bag_of_words = parseTweet(row[1])
        train_bag_of_words.append(bag_of_words)
    train_bag_of_words = np.array(train_bag_of_words)

    printInfo("Parsing test data")

    # Parse testing data
    test_data = parseTwitterCSV(test_csv)
    test_header = test_data[0]
    test_data = test_data[1:]

    # Create test bag of words
    test_bag_of_words = []
    for row in test_data:
        bag_of_words = parseTweet(row[1])
        test_bag_of_words.append(bag_of_words)
    test_bag_of_words = np.array(test_bag_of_words)

    printInfo("Determining top unigrams")

    # Features consider the top m words of both the testing and training data
    all_bags = np.concatenate((train_bag_of_words,test_bag_of_words),axis=0)

    # 'fit' feature generator on all bag of words 
    feat_generator = TopWordsFeatures(m=num_feats)
    feat_generator.fit(all_bags)
    del all_bags

    printInfo("Generating training features")
    # Generate features for train
    X_train = feat_generator.generateFeatures(train_bag_of_words)
    del train_bag_of_words

    printInfo("Generating test features")
    # Generate features for test
    X_test = feat_generator.generateFeatures(test_bag_of_words)
    del test_bag_of_words

    # Predicted scores for each variable
    y_pred = {}
    for var_name in var_names:
        printInfo("Processing var '%s'" % (var_name,))
        assert var_name in train_header
        var_index = train_header.index(var_name)

        # Compose y train
        y_train = []
        for row in train_data:
            y_train.append(np.float(row[var_index]))
        y_train = np.array(y_train)

        # Initialize classifier
        # clf = Ridge(alpha=1e-4)
        # clf = ElasticNet(alpha=1e-5);
        clf = svm.SVR()

        # Train classifier
        printInfo("  Training classifier")
        clf.fit(X_train,y_train)

        # Predict
        printInfo("  Running prediction")
        exp_num_values = X_test.shape[0]
        y_pred[var_name] = clf.predict(X_test)

        greater_than_1 = 0
        less_than_0 = 0

        printInfo("  Correcting predictions")

        # Because scores can only be between 0.0 and 1.0 if the classifier
        # predicts a score to be out of that range then that score must be
        # brought back within range. Counts are kept and displayed to give a
        # sense of if the classifier is working reasonably.
        for i in range(len(y_pred[var_name])):
            if y_pred[var_name][i] < 0.0:
                y_pred[var_name][i] = 0.0
                less_than_0 += 1
            elif y_pred[var_name][i] > 1.0:
                y_pred[var_name][i] = 1.0
                greater_than_1 += 1

        assert len(y_pred[var_name]) == exp_num_values

        printInfo("    Scores > 1.0: %s" % (greater_than_1,))
        printInfo("    Scores < 0.0: %s" % (less_than_0,))

    # Open submission file and write out results
    try:
        out_csv = open(sys.argv[3], 'w')
    except IOError:
        sys.stderr.write("[Error] Could not open file '%s'" % (sys.argv[3],))
        sys.exit(2)

    printInfo("Writing predictions to submission file")

    # Write header
    out_csv.write('id,s1,s2,s3,s4,s5,w1,w2,w3,w4,k1,k2,k3,k4,k5,k6,k7,k8,k9,k10,k11,k12,k13,k14,k15')
    for i in range(len(test_data)):
        row_id = test_data[i][0]
        # Write one row at a time
        out_csv.write('\n%s' % (row_id,))
        for var_name in var_names:
            out_csv.write(',%s' % (y_pred[var_name][i],))



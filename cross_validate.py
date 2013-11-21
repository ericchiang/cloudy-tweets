#!/usr/bin/python

__author__    = "Eric Chiang"
__copyright__ = "Copyright 2013, Eric Chiang"
__email__     = "eric.chiang.m@gmail.com"

__license__   = "GPL"
__version__   = "3.0"

import sys
import numpy as np
import pickle as pk
from sklearn.cross_validation import KFold
from sklearn.linear_model import *
from sklearn import svm
from sklearn.decomposition import PCA
from feature_generation import *
from format_data import *
from utils import *

"""
This file handles the classification pipeline; reading the train.csv file,
generating feature matrix, and preforming cross validation.
"""

#Run cross validation.
def runCV(y,k_fold):
    y_pred = np.array([0.] * len(y))
    i = 0
    for train_indices,test_indices in k_fold:
        # clf = svm.SVR()
        clf = ElasticNet(alpha=1e-5)
        # clf = Ridge(alpha=1e-7) # Ridge Regression... really fast

        printInfo("  Fold %s: fitting" % (i + 1,))
        # Read training data from file
        train_file = open('fold_data/train_%s.pk' % (i,), 'r')
        X_train = pk.load(train_file)
        train_file.close()
        y_train = y[train_indices]
        # fit model
        clf.fit(X_train,y_train)
        # Clear Memory
        del X_train
        del y_train

        printInfo("  Fold %s: predicting" % (i + 1,))
        # Read test data from file
        test_file = open('fold_data/test_%s.pk' % (i,), 'r')
        X_test  = pk.load(test_file)
        test_file.close()
        # Run prediction
        y_pred[test_indices] = clf.predict(X_test)
        # Clear memory
        del clf
        del X_test
        i += 1

    # Because all scores must be between 0 and 1, augment predicted scores 
    for i in range(len(y_pred)):
        if y_pred[i] < 0.0:
            y_pred[i] = 0.0
        elif y_pred[i] > 1.0:
            y_pred[i] = 1.0
    print ' '
    return y_pred

"""
Print usage
"""
def usage():
    print """
        Usage:
        %s [train.csv] [num features]
        """ % (sys.argv[0],)


"""
Do everything
"""
if __name__ == '__main__':
    if len(sys.argv) != 3:
        usage()
        sys.exit(2)

    # Attempt to open training file
    try:
        tweet_csv = open(sys.argv[1], 'rb')
    except IOError:
        sys.stderr.write("[Error] Could not open file '%s'" % (sys.argv[1],))
        sys.exit(2)

    num_feats = int(sys.argv[2])

    # Variables cv should be run on
    var_names = ['s1','s2','s3','s4','s5',
                 'w1','w2','w3','w4',
                 'k1','k2','k3','k4','k5','k6','k7','k8',
                 'k9','k10','k11','k12','k13','k14','k15']

    # Parse data from file
    data = parseTwitterCSV(tweet_csv)
    headers = data[0]
    data = data[1:]

    # Read features from file or generate new ones?
    use_existing_features = False

    if not use_existing_features:
        # Parse each tweet into a bag of words
        X_bag_of_words = []
        for row in data:
            bag_of_words = parseTweet(row[1])
            X_bag_of_words.append(bag_of_words)
        X_bag_of_words = np.array(X_bag_of_words)
    
    
        printInfo("Determining features space")
        feat_generator = TopWordsFeatures(m=num_feats)
        # Determine the top m words and define the feature space
        feat_generator.fit(X_bag_of_words)

        """
        Generate folds and save them to file. Because spliting a sparce matrix
        into training and test takes a monstrous amount of time, the bag of 
        words are split and used to generate training and test sparce matrices
        directly.
        """
        num_folds = 3
        printInfo("Saving folds to disk (%s folds)" % (num_folds,))
        k_fold = KFold(n=len(X_bag_of_words),n_folds=num_folds,indices=True)
        i = 0
        for train_indices,test_indices in k_fold:

            printInfo("  %s Processing sparce training features" % (i,))
            # Generate trianing fold from bag of words
            train_bag_of_words = X_bag_of_words[train_indices]
            # Map bag of words onto feature space
            train_sparce_feats = \
                    feat_generator.vectorize(train_bag_of_words)
            del train_bag_of_words

            # Save training fold to file using pickle
            train_file = open('fold_data/train_%s.pk' % (i,),'w')
            pk.dump(train_sparce_feats,train_file)
            del train_sparce_feats
            train_file.close()

            printInfo("  %s Processing sparce test features" % (i,))
            # Generate test fold from bag of words
            test_bag_of_words = X_bag_of_words[test_indices]
            # Map bag of words onto feature space
            test_sparce_feats = \
                   feat_generator.vectorize(test_bag_of_words)
            del test_bag_of_words

            # Save test fold to file using pickle
            test_file = open('fold_data/test_%s.pk' % (i,),'w')
            pk.dump(test_sparce_feats,test_file)
            del test_sparce_feats
            test_file.close()

            # i++
            i += 1
   
        # Save k folds object to file for future runs 
        folds_file = open('fold_data/k_fold.pk','w')
        pk.dump(k_fold,folds_file)
        folds_file.close()

    # If folds are already generated then the K folds object must be recalled
    else:
        folds_file = open('fold_data/k_fold.pk','r')
        k_fold = pk.load(folds_file)
        folds_file.close()

    printInfo("Running cross validation")

    mses = []
    # Run cross validation on each variable
    for var_name in var_names:
        if var_name not in headers:
            printInfo("No variable named '%s'" % (var_name,))
            continue
        var_index = headers.index(var_name)

        # Generate target data
        y = []
        for row in data:
            y.append(np.float(row[var_index]))
        y = np.array(y)

        # Run cross validation
        printInfo("  Running CV on vairable '%s'" % (var_name,))
        y_pred = runCV(y,k_fold)

        # Calculate mean squared error
        mse = np.mean((y - y_pred)**2)
        del y_pred
        mses.append(mse)
        printInfo("  Results for variable '%s'" % (var_name,))
        printInfo("    MSE : %s" % (mse,))

    # Print all mean squared error cleanly (without timestamp)
    for mse in mses:
        print mse

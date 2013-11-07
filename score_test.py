#!/usr/bin/python

import sys
import numpy as np
import pickle as pk
from sklearn.cross_validation import KFold
from sklearn.linear_model import Ridge
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

    num_feats = 3000

    var_names = ['s1','s2','s3','s4','s5',
                 'w1','w2','w3','w4',
                 'k1','k2','k3','k4','k5','k6','k7','k8',
                 'k9','k10','k11','k12','k13','k14','k15']

    train_data = parseTwitterCSV(train_csv)
    train_header = train_data[0]
    train_data = train_data[1:]

    train_bag_of_words = []
    for row in train_data:
        bag_of_words = parseTweet(row[1])
        train_bag_of_words.append(bag_of_words)
    train_bag_of_words = np.array(train_bag_of_words)


    test_data = parseTwitterCSV(test_csv)
    test_header = test_data[0]
    test_data = test_data[1:]

    test_bag_of_words = []
    for row in test_data:
        bag_of_words = parseTweet(row[1])
        test_bag_of_words.append(bag_of_words)
    test_bag_of_words = np.array(test_bag_of_words)

    all_bags = np.concatenate((train_bag_of_words,test_bag_of_words),axis=0)

    feat_generator = TopWordsFeatures(m=num_feats)
    feat_generator.fit(all_bags)
    del all_bags

    printInfo("Generating training features")
    X_train = feat_generator.generateFeatures(train_bag_of_words)
    del train_bag_of_words

    """
    train_file = open('fold_data/train_final.pk','w')
    pk.dump(X_train,train_file)
    del X_train
    train_file.close()
    """

    printInfo("Generating test features")
    X_test = feat_generator.generateFeatures(test_bag_of_words)
    del test_bag_of_words

    """
    test_file = open('fold_data/test_final.pk','w')
    pk.dump(X_test,test_file)
    del X_test
    test_file.close()
    """

    y_pred = {}
    for var_name in var_names:
        printInfo("Processing var '%s'" % (var_name,))
        if var_name not in train_header:
            continue
        var_index = train_header.index(var_name)

        y_train = []
        for row in train_data:
            y_train.append(np.float(row[var_index]))
        y_train = np.array(y_train)

        clf = Ridge(alpha=1e-7)

        """
        train_file = open('fold_data/train_final.pk','r')
        X_train = pk.load(train_file)
        train_file.close()
        """

        printInfo("  Training classifier")
        clf.fit(X_train,y_train)

        # del X_train
        # del y_train

        """
        test_file = open('fold_data/test_final.pk','r')
        X_test = pk.load(test_file)
        test_file.close()
        """

        printInfo("  Running prediction")

        exp_num_values = X_test.shape[0]
        y_pred[var_name] = clf.predict(X_test)

        for i in range(len(y_pred[var_name])):
            if y_pred[var_name][i] < 0.0:
                y_pred[var_name][i] = 0.0
            elif y_pred[var_name][i] > 1.0:
                y_pred[var_name][i] = 1.0

        # del X_test
        # del clf
        assert len(y_pred[var_name]) == exp_num_values

    try:
        out_csv = open(sys.argv[3], 'w')
    except IOError:
        sys.stderr.write("[Error] Could not open file '%s'" % (sys.argv[3],))
        sys.exit(2)

    out_csv.write('id,s1,s2,s3,s4,s5,w1,w2,w3,w4,k1,k2,k3,k4,k5,k6,k7,k8,k9,k10,k11,k12,k13,k14,k15')
    for i in range(len(test_data)):
        row_id = test_data[i][0]
        out_csv.write('\n%s' % (row_id,))
        for var_name in var_names:
            out_csv.write(',%s' % (y_pred[var_name][i],))



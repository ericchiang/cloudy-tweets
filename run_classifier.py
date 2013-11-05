import sys
import numpy as np
import pickle as pk
from sklearn.cross_validation import KFold
from sklearn.linear_model import *
from sklearn.decomposition import PCA
from feature_generation import *
from format_data import *
from utils import *

"""
Run cross validation
"""
def runCV(y,k_fold):
    y_pred = np.array([0.] * len(y))
    i = 0
    for train_indices,test_indices in k_fold:
        clf = Ridge(alpha=1e-7)
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

    # Because all scores must be between 0 and 1, augment 
    for i in range(len(y_pred)):
        if y_pred[i] < 0.0:
            y_pred[i] = 0.0
        elif y_pred[i] > 1.0:
            y_pred[i] = 1.0
    return y_pred

"""
Print usage
"""
def usage():
    print """
        Usage:
        %s [train.csv] [var names... ]
        """ % (sys.argv[0],)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        usage()
        sys.exit(2)

    # Attempt to open training file
    try:
        tweet_csv = open(sys.argv[1], 'rb')
    except IOError:
        sys.stderr.write("[Error] Could not open file '%s'" % (sys.argv[1],))
        sys.exit(2)

    # Variables cv should be run on
    if len(sys.argv) > 2:
        var_names = sys.argv[2:]
    else:
        var_names = ['s1','s2','s3','s4','s5',
                     'w1','w2','w3','w4',
                     'k1','k2','k3','k4','k5','k6','k7','k8',
                     'k9','k10','k11','k12','k13','k14','k15']

    # Parse data from file
    data = parseTwitterCSV(tweet_csv)
    headers = data[0]
    data = data[1:]

    use_existing_features = True

    if not use_existing_features:
        # Parse each tweet into a bag of words
        X_bag_of_words = []
        for row in data:
            bag_of_words = parseTweet(row[1])
            X_bag_of_words.append(bag_of_words)
        X_bag_of_words = np.array(X_bag_of_words)
    
    
        """
        Generate feature matrix using the top n words. Each column represents a
        word and each value represents the count of said word for that row.
        """
        printInfo("Generating Features")
        feat_generator = TopWordsFeatures(n=600)
        X = feat_generator.fit(X_bag_of_words).generateFeatures(X_bag_of_words)
        del X_bag_of_words
    
        printInfo("Features generated, shape: %s" % (X.shape,))
    
        num_folds = 10
        printInfo("Saving folds to disk (%s folds)" % (num_folds,))
        k_fold = KFold(n=X.shape[0],n_folds=num_folds,indices=True)
        i = 0
        for train_indices,test_indices in k_fold:
            printInfo("  Processing fold %s" % (i))
            train_file = open('fold_data/train_%s.pk' % (i,),'w')
            pk.dump(X[train_indices,:],train_file)
            train_file.close()
            test_file = open('fold_data/test_%s.pk' % (i,),'w')
            pk.dump(X[test_indices,:],test_file)
            test_file.close()
            i += 1
    
        del X
        folds_file = open('fold_data/k_fold.pk','w')
        pk.dump(k_fold,folds_file)
        folds_file.close()
    else:
        folds_file = open('fold_data/k_fold.pk','r')
        k_fold = pk.load(folds_file)
        folds_file.close()


    """

    pca = PCA()
    pca.fit(X)

    pca.n_components = 220

    X = pca.fit_transform(X)
    """
    # printInfo("Feature reduced, shape: %s" % (X.shape,))

    printInfo("Running cross validation")

    mses = []
    for var_name in var_names:
        if var_name not in headers:
            printInfo("No variable named '%s'" % (var_name,))
            continue
        var_index = headers.index(var_name)

        y = []
        for row in data:
            y.append(np.float(row[var_index]))
        y = np.array(y)

        printInfo("  Running CV on vairable '%s'" % (var_name,))
        y_pred = runCV(y,k_fold)
        mse = np.mean((y - y_pred)**2)
        del y_pred
        mses.append(mse)
        printInfo("  Results for variable '%s'" % (var_name,))
        printInfo("    MSE : %s" % (mse,))
    for mse in mses:
        print mse

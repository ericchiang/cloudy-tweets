import sys
import numpy
from sklearn.cross_validation import KFold
from sklearn.linear_model import *
from sklearn.decomposition import PCA
from feature_generation import *
from format_data import *
from utils import *

"""
Run cross validation
"""
def runCV(X,y,num_folds=10):
    k_fold = KFold(n=len(X),n_folds=num_folds,indices=True)
    y_pred = numpy.array([0.] * len(y))
    for train_indices,test_indices in k_fold:
        X_train = X[train_indices]
        y_train = y[train_indices]
        X_test  = X[test_indices]
        y_test  = y[test_indices]
        clf     = Ridge(alpha=1e-5) # Run Ridge Regression
        y_pred[test_indices] = clf.fit(X_train,y_train).predict(X_test)
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

    # Parse each tweet into a bag of words
    X_bag_of_words = []
    for row in data:
        bag_of_words = parseTweet(row[1])
        X_bag_of_words.append(bag_of_words)
    X_bag_of_words = numpy.array(X_bag_of_words)


    """
    Generate feature matrix using the top n words. Each column represents a
    word and each value represents the count of said word for that row.
    """
    feat_generator = TopWordsFeatures(n=270)
    X = feat_generator.fit(X_bag_of_words).generateFeatures(X_bag_of_words)

    printInfo("Features generated, shape: %s" % (X.shape,))

    """

    pca = PCA()
    pca.fit(X)

    pca.n_components = 220

    X = pca.fit_transform(X)
    """
    # printInfo("Feature reduced, shape: %s" % (X.shape,))

    mses = []
    for var_name in var_names:
        if var_name not in headers:
            printInfo("No variable named '%s'" % (var_name,))
            continue
        var_index = headers.index(var_name)

        y = []
        for row in data:
            y.append(numpy.float(row[var_index]))
        y = numpy.array(y)

        y_pred = runCV(X,y)
        mse = numpy.mean((y - y_pred)**2)
        mses.append(mse)
        rmse = numpy.sqrt(mse)
        printInfo("Results for variable '%s'" % (var_name,))
        printInfo("  MSE : %s" % (mse,))
        printInfo("  RMSE: %s" % (rmse,))
    for mse in mses:
        print mse

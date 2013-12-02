#!/usr/bin/python

__author__    = "Eric Chiang"
__copyright__ = "Copyright 2013, Eric Chiang"
__email__     = "eric.chiang.m@gmail.com"

__license__   = "GPL"
__version__   = "3.0"

from Queue import Queue
from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVR
from threading import Thread
from time import sleep
from tweet_text import TweetTokenizer
import numpy as np
import pandas as pd
import sys

"""
Cross validation file
"""


def run_fold(train_indices,test_indices,vectorizer,train_data,pred,queue):
    """ Run fold computation in parallel
    """
    print "Processing Fold"
    train_raw = raw_tweets[train_indices].tolist()
    test_raw  = raw_tweets[test_indices].tolist()
    X_train = vectorizer.transform(train_raw)
    X_test  = vectorizer.transform(test_raw)
    for variable in variables:
        y_train = np.array(train_data[variable])[train_indices]
        clf = SVR()
        clf.fit(X_train,y_train)
        pred[variable][test_indices] = clf.predict(X_test)
    # When finished notify queue
    queue.put("done")

# Load train data from file 
train_data = pd.read_csv(open('data/train.csv'),quotechar='"')

# Numpy array used to allow for test/train splitting
raw_tweets = np.array(train_data['tweet'])
variables = train_data.columns[4:].tolist()

if len(variables) != 24:
   sys.stderr.write(variables)
   raise Exception("Bad number of sentiments")

# Allocate data frame for predictions
pred = train_data.copy()

# Build vectorizer
tokenizer = TweetTokenizer()
print "Building vectorizer"
vectorizer = CountVectorizer(tokenizer=tokenizer.tokenize_tweet,
                                       max_features=50,
                                       binary=True,
                                       ngram_range=(1,1))

# Vectorization is done after folds are split. It's takes more time to slice a
# sparce matrix than a list of strings.
vectorizer.fit(raw_tweets)


q = Queue()
num_threads = 4
num_folds = 8
threads = []

# Spawn a thread for each fold
k_fold = KFold(n=len(raw_tweets),n_folds=num_folds,indices=True)
for train_indices,test_indices in k_fold:
    if len(threads) >= num_threads:
        while q.empty():
            sleep(2)
        status = q.get()
    t = Thread(target=run_fold, args=(train_indices,test_indices,
                                      vectorizer,train_data,pred,q))
    t.deamon = True
    t.start()
    threads.append(t)

# Wait for all threads to complete
for t in threads:
    t.join()

# Calculate (and print) mean squred errors for each variable
print "Mean Squared Errors:"
for variable in variables:
    y_pred = np.array(pred[variable])
    y_actu = np.array(train_data[variable])

    mse = np.average((y_pred - y_actu)**2)
    print "'%s': %f" % (variable,mse)

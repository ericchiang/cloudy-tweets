

from sklearn.cross_validation import KFold
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVR
import nltk
import numpy as np
import pandas as pd
import sys

train_data = pd.read_csv(open('data/train.csv'),quotechar='"')
# Numpy array used to allow for test/train splitting
raw_tweets = np.array(train_data['tweet'])
sentiments = train_data.columns[4:].tolist()

if len(sentiments) != 24:
   sys.stderr.write(sentiments)
   raise Exception("Bad number of sentiments")

pred = train_data.copy()

print "Building vectorizer"
vectorizer = CountVectorizer(tokenizer=nltk.word_tokenize,
                                       max_features=3000,
                                       binary=True,
                                       ngram_range=(1,1))

vectorizer.fit(raw_tweets)

num_folds = 6
k_fold = KFold(n=len(raw_tweets),n_folds=num_folds,indices=True)

for train_indices,test_indices in k_fold:
    print "Processing Fold"
    train_raw = raw_tweets[train_indices].tolist()
    test_raw  = raw_tweets[test_indices].tolist()
    X_train = vectorizer.transform(train_raw)
    X_test  = vectorizer.transform(test_raw)
    for sentiment in sentiments:
        y_train = np.array(train_data[sentiment])[train_indices]
        clf = ElasticNet(alpha=1e-5)
        clf.fit(X_train,y_train)
        pred[sentiment][test_indices] = clf.predict(X_test)

print "Mean Squared Errors:"
for sentiment in sentiments:
    y_pred = np.array(pred[sentiment])
    y_actu = np.array(train_data[sentiment])

    # Compute mean squared error
    mse = np.average((y_pred - y_actu)**2)
    print "'%s': %f" % (sentiment,mse)

#!/usr/bin/python

__author__    = "Eric Chiang"
__copyright__ = "Copyright 2013, Eric Chiang"
__email__     = "eric.chiang.m@gmail.com"

__license__   = "GPL"
__version__   = "3.0"


from account import username,apikey
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVR
from yhat import Yhat,BaseModel

import nltk
import numpy as np
import pandas as pd
import sys

class TweetClassifier(BaseModel):
    """Yhat classifier to be uploaded to server.
    """
    def transform(self,raw_tweets):
        import nltk
        return self.vectorizer.transform(raw_tweets)
    
    def predict(self,x):
        pred = np.array(self.clf.predict(x))
        pred[np.where(pred > 1.0)] = 1.0
        pred[np.where(pred < 0.0)] = 0.0
        return {"scores" : pred}

train_data = pd.read_csv(open('data/train.csv','r'),quotechar='"')

raw_tweets = train_data['tweet'].tolist()
sanity_raw = raw_tweets[:100]

sentiments = train_data.columns[4:].tolist()
vectorizer = CountVectorizer(tokenizer=nltk.word_tokenize,
                             stop_words='english',
                             max_features=3000,
                             binary=True,
                             ngram_range=(1,1))

yh = Yhat(username,apikey)

X_train = vectorizer.fit_transform(raw_tweets)

for sentiment in sentiments:
    print "Processing '%s'" % sentiment
    clf = SVR()
    y_train = train_data[sentiment].tolist()

    print "Training classifier"
    clf.fit(X_train,y_train)

    tweet_clf = TweetClassifier(clf=clf,vectorizer=vectorizer)
    model_name = "TweetClassifier_%s" % (sentiment,)

    print "Uploading to yhat"
    upload_status = yh.upload(model_name,tweet_clf)
    model_version = upload_status['version'] 

    print "'%s':'%s' uploaded to yhat" % (model_name,model_version)

    # Sanity check uploaded classifier by comparing remote against local scores

    print "Preforming sanity check"
    print "Predicting local scores"
    local_sanity = tweet_clf.predict(tweet_clf.transform(sanity_raw))['scores']
    local_sanity = np.array(local_sanity)

    print "Getting scores from server"
    results_from_server = yh.raw_predict(model_name,model_version,sanity_raw)
    try:
        server_sanity = results_from_server['prediction']['scores']
    except:
        print results_from_server
        sys.exit(3)
    server_sanity = np.array(server_sanity)

    # Because of float point scores compare difference of scores to some level
    # of tolerance rather than checking equality
    score_diff = np.abs(local_sanity - server_sanity)

    sanity_tolerance = 1e-3
    sanity_status = np.alltrue(score_diff < sanity_tolerance)

    if not sanity_status:
        sys.stderr.write("Sanity check failed\n")
        sys.stderr.write("Local sanity scores\n%s\n" % (local_sanity,))
        sys.stderr.write("Server sanity scores\n%s\n" % (server_sanity,))
        raise Exception("Sanity check failed")

    print "Sanity check passed" 

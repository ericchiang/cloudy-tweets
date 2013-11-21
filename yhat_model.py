#!/usr/bin/python

__author__    = "Eric Chiang"
__copyright__ = "Copyright 2013, Eric Chiang"
__email__     = "eric.chiang.m@gmail.com"

__license__   = "GPL"
__version__   = "3.0"

"""
Create a yhat model
"""

import sys
import nltk
import numpy as np
import pandas as pd
from account import __username__, __apikey__
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import Ridge
from utils import *
from yhat import Yhat, BaseModel


# Meta-data
train_file = 'data/train.csv'
test_file  = 'data/test.csv'
vars = ['s1','s2','s3','s4','s5',
        'w1','w2','w3','w4',
        'k1','k2','k3','k4','k5','k6','k7','k8',
        'k9','k10','k11','k12','k13','k14','k15']

class CloudyClassifier(BaseModel):
    def require(self):
        import nltk

    def transform(self, raw):
        X = self.vectorizer.transform(raw)
        return X

    def predict(self,X):
        pred = self.clf.predict(X)
        return {"scores" : pred}

print "Parsing data"
train_data = pd.read_csv(open(train_file),quotechar='"')[:10000]
test_data  = pd.read_csv(open(test_file),quotechar='"')[:100]

print "Vectorizing raw tweets"
# default of 100 words
vectorizer = CountVectorizer(tokenizer=nltk.word_tokenize,
                             max_features=300,
                             ngram_range=(1,1,))

X_train = vectorizer.fit_transform(np.array(train_data['tweet']))

y_train = np.array(train_data['w2'])

print "Training classifier"

clf = Ridge(alpha=1e-5)
clf.fit(X_train,y_train)

print "yhat!"
print __username__
print __apikey__

model_name = "CloudyClassifier"

yclf = CloudyClassifier(vectorizer=vectorizer,clf=clf)

yh = Yhat(__username__,__apikey__)

print "Predicting local"

transformed = yclf.transform(test_data['tweet'].tolist())
results = yclf.predict(transformed)

print "Uploading to yhat"
upload_status = yh.upload(model_name,yclf)
print upload_status
model_version = upload_status['version']

results_from_server = yh.raw_predict(model_name,
                                     model_version,
                                     test_data['tweet'].tolist())
print results['scores']
print results_from_server['prediction']['scores']

print 'results all match => %s' \
    % np.all(np.array(results['scores']) == \
              np.array(results_from_server['prediction']['scores']))

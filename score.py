#!/usr/bin/python

__author__    = "Eric Chiang"
__copyright__ = "Copyright 2013, Eric Chiang"
__email__     = "eric.chiang.m@gmail.com"

__license__   = "GPL"
__version__   = "3.0"

from account_info import username,apikey
from yhat import Yhat
import pandas as pd
import numpy as np
import sys

"""
Score train data using yhat and create a submission
"""

if len(sys.argv) != 2:
    sys.stderr.write("Please specify a file to write predictions out to!\n")
    sys.exit(2)

sub_file = sys.argv[1]

# Record the best model
best_model =\
{
 's1':1, 's2':1, 's3':1, 's4':1, 's5':1, 
 'w1':1, 'w2':1, 'w3':1, 'w4':1,
 'k1':1, 'k2':1, 'k3':1, 'k4':1, 'k5':1, 
 'k6':1, 'k7':1, 'k8':1, 'k9':1, 'k10':1, 
 'k11':1, 'k12':1, 'k13':1, 'k14':1, 'k15':1
}

test_data = pd.read_csv(open('data/test.csv','r'),quotechar='"')

sub_data = pd.read_csv(open('data/sampleSubmission.csv','r'),quotechar='"')

if not np.alltrue(test_data['id'] == sub_data['id']):
    raise Exception("IDs do not match")

yh = Yhat(username, apikey)

variabless = sub_data.columns[1:]
raw_tweets = test_data['tweet'].tolist()

for variable in variables:
    model_version = best_model[variable]
    model_name = "TweetClassifier_%s" % (variable,)
    results_from_server = yh.raw_predict(model_name,
                                         model_version,
                                         raw_tweets)
    pred = results_from_server['prediction']['scores']
    sub_data[variable] = pred

try:
    sub_data.to_csv(open(sub_file,'w'),index=False)
except IOError:
    sys.stderr.write("IO error: could not write data to file")

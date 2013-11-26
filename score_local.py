
from Queue import Queue
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVR
from threading import Thread
from time import sleep
from tweet_text import TweetTokenizer
import pandas as pd
import numpy as np
import sys


def score_variable(variable,queue):
    print "Scoring: '%s'" % variable
    clf = SVR()
    y_train = np.array(train_data[variable])
    clf.fit(X_train,y_train)
    y_pred = np.array(clf.predict(X_test))
    y_pred[np.where(y_pred > 1.0)] = 1.0
    y_pred[np.where(y_pred < 0.0)] = 0.0
    sub_data[variable] = y_pred
    queue.put("done")

if len(sys.argv) != 2:
    sys.stderr.write("Please specify a file to write predictions out to!\n")
    sys.exit(2)

sub_file = sys.argv[1]

train_data = pd.read_csv(open('data/train.csv','r'),quotechar='"')
test_data = pd.read_csv(open('data/test.csv','r'),quotechar='"')
sub_data = pd.read_csv(open('data/sampleSubmission.csv','r'),quotechar='"')


if not np.alltrue(test_data['id'] == sub_data['id']):
    raise Exception("IDs do not match")

train_tweets = train_data['tweet'].tolist()
tokenizer = TweetTokenizer()
test_tweets = test_data['tweet'].tolist()
vectorizer = CountVectorizer(tokenizer=tokenizer.tokenize_tweet,
                             max_features=3000,
                             binary=True,
                             ngram_range=(1,1))

print "Training vectorizer"
vectorizer.fit(train_tweets + test_tweets)

print "Transforming tweets to vector space"
X_train = vectorizer.transform(train_tweets)
X_test = vectorizer.transform(test_tweets)
variables = sub_data.columns[1:]


q = Queue()
num_threads = 4
threads = []
for variable in variables:
    if len(threads) >= num_threads:
        while q.empty():
            sleep(5)
        status = q.get()
    t = Thread(target=score_variable, args=(variable,q))
    t.deamon = True
    t.start()
    threads.append(t)

for t in threads:
    t.join()

try:
    sub_data.to_csv(open(sub_file,'w'),index=False)
except IOError:
    sys.stderr.write("IO error: could not write data to file")

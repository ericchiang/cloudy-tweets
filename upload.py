
from account import username,apikey
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVR
from tweet_text import TweetTokenizer
from yhat import Yhat, BaseModel
import numpy as np
import pandas as pd

sanity_tolerance = 0.001


class CloudyClassifier(BaseModel):
    def require(self):
        import re
        import string
        self.vectorizer.tokenizer=self.tokenizer.tokenize_tweet


    def transform(self, raw):
        X = self.vectorizer.transform(raw)
        return X

    def predict(self,X):
        pred = np.array(self.clf.predict(X))
        pred[np.where(pred > 1.0)] = 1.0
        pred[np.where(pred < 0.0)] = 0.0
        return {"scores" : pred}

train_data = pd.read_csv(open('data/train.csv','r'),quotechar='"')[:1000]

raw_tweets = train_data['tweet'].tolist()
sanity_raw = raw_tweets[:100]


sentiments = train_data.columns[4:].tolist()
tokenizer = TweetTokenizer()
vectorizer = CountVectorizer(tokenizer=tokenizer.tokenize_tweet,
                             max_features=3000,
                             binary=True,
                             ngram_range=(1,1))

yh = Yhat(username,apikey)

X_train = vectorizer.fit_transform(raw_tweets)
vectorizer.tokenizer=None

for sentiment in sentiments:
    print "Processing '%s'" % sentiment
    clf = SVR()
    y_train = train_data[sentiment].tolist()

    print "  Training classifier"
    clf.fit(X_train,y_train)

    tweet_clf = CloudyClassifier(clf=clf,vectorizer=vectorizer,tokenizer=tokenizer)
    model_name = "TweetClassifier-%s" % (sentiment,)

    print "  Uploading to yhat"
    upload_status = yh.upload(model_name,tweet_clf)
    print upload_status
    model_version = upload_status['version'] 

    print "  '%s':'%s' uploaded to yhat" % (model_name,model_version)

    print "  Preforming sanity check"
    print "    Predicting local scores"
    sanity_transformed = tweet_clf.transform(sanity_raw)
    local_sanity = tweet_clf.predict(sanity_transformed)['scores']
    local_sanity = np.array(local_sanity)

    print "    Getting scores from server"
    results_from_server = yh.raw_predict(model_name,model_version,sanity_raw)
    server_sanity = results_from_server['prediction']['scores']
    server_sanity = np.array(server_sanity)

    # Because of float point scores compare difference of scores to some level
    # of tolerance rather than checking equality
    score_diff = np.abs(local_sanity - server_sanity)
    sanity_status = np.alltrue(score_diff < sanity_tolerance)

    if not sanity_status:
        sys.stderr.write("Sanity check failed\n")
        sys.stderr.write("Local sanity scores\n%s\n" % (local_sanity,))
        sys.stderr.write("Server sanity scores\n%s\n" % (server_sanity,))
        raise Exception("Sanity check failed")

    print "  Sanity check passed"

 

import re
import string

"""
Built class to hopefully make this easier on yhat
"""
class TweetTokenizer(object):
    def __init__(self):
        self.re_number = re.compile(r'^\d*\.?\d+$')
        self.re_alphanum = re.compile('\d')
        self.re_weather = re.compile("#WEATHER.* (\d+\.\d+)F.* (\d+\.\d+)% Humidity. (\d+\.\d+)MPH")
        self.re_time = re.compile(r'\d{1,2}:\d\d[ ]?(am|pm)?')       # Time
        self.re_temp = re.compile(r'(\d+\.?\d*) ?(fahrenheit|celcius|f|c|degrees|degree)(\W|$)')
        self.re_velo = re.compile(r'\d+\.?\d* ?mph')                 # Velocity
        self.re_perc = re.compile(r'\d+\.?\d* ?(%|percent)')
        self.re_punc = re.compile(r'[%s]' % re.escape(string.punctuation))
        self.re_numb = re.compile(r'^-?\d+\.?\d*$')        
        
        self.pos_emoticons = [':)',':-)',' : )',':D','=)',' : D ','(:','(=']
        self.neg_emoticons = [':(',':-(',' : (','=(','):',')=']
        self.ntr_emoticons = [':/','=/',':\\','=\\',':S','=S',':|','=|']
        
        self.meta_mention = '@mention'
        self.meta_link = '{link}'

    def tokenize_tweet(self,tweet):

        m = self.re_weather.match(tweet)
        if m:
            temp = float(m.group(1))
            humd = float(m.group(2))
            mph  = float(m.group(3))
            sent = ""
            if temp > 85:
                sent = 'HOTNUMBER'
            elif temp > 45:
                sent = 'NICENUMBER'
            else:
                sent = 'COLDNUMBER'
    
            temp = str(int(temp / 10.0) * 10)
            humd = str(int(humd / 10.0) * 10)
            mph  = str(int(mph / 10.0) * 10)
            tokens = ['WEATHER','MPH'+mph,'TEMP'+temp,'HUMD'+humd,sent, 'TEMP']
            return tokens
    
        tweet = tweet.lower()
    
        if '!' in tweet:
            tweet = tweet.replace('!',' EXL ')
    
        if '?' in tweet:
            tweet = tweet.replace('?',' QST ')
    
        for emoticon in self.pos_emoticons:
            if emoticon in tweet:
                tweet = tweet.replace(emoticon,' SMILEY ')
    
        for emoticon in self.neg_emoticons:
            if emoticon in tweet:
                tweet = tweet.replace(emoticon,' FROWNY ')
    
        if ';)' in tweet:
            tweet = tweet.replace(';)',' WINKY ')
    
        if self.meta_mention in tweet:
            tweet = tweet.replace(self.meta_mention,' MENTION ')
    
        if self.meta_link in tweet:
            tweet = tweet.replace(self.meta_link,' LINK ')
    
        tweet = self.re_time.sub(' TIME ',tweet)
        tweet = self.re_temp.sub(r' TEMP \1 ',tweet)     
        tweet = self.re_velo.sub(r' WIND ',tweet)
        tweet = self.re_perc.sub(r' PERC ',tweet)
    
        tokens = tweet.split()
    
        return_tokens = []
    
        for token in tokens:
            token = self.re_punc.sub('',token)
            if self.re_numb.match(token):
                token = float(token)
                if token > 120:
                    token = 'LARGENUMBER'
                elif token > 85:
                    token = 'HOTNUMBER'
                elif token > 45:
                    token = 'NICENUMBER'
                elif token > 10:
                    token = 'COLDNUMBER'
                else:
                    token = 'SMALLNUMBER'
            if token:
                return_tokens.append(token)
    
        return return_tokens

# Test 
if __name__ == '__main__':
    import pandas as pd
    tokenizer = TweetTokenizer()
    train_data = pd.read_csv(open('data/train.csv','r'),quotechar='"')
    tweets = train_data['tweet'].tolist()
    for tweet in tweets[:100]:
        print tweet
        print " ".join(tokenizer.tokenize_tweet(tweet))

import re
import string
import nltk


class TweetTokenizer(Object):

    def __init__:
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


    def build_tokenizer():
        return self.tokenize_tweet

    def tokenize_tweet(tweet):
        tweet = tweet.lower()

        weather_match = self.re_weather.match(tweet):
        if weather_match:
            temp = float(m.group(1))
            humd = float(m.group(2))

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
            if self.re_numb.match(token):
                token = float(token)
                if token > 120:
                    token = 'LARGE NUMBER'
                elif token > 85:
                    token = 'HOT NUMBER'
                elif token > 45:
                    token = 'NICE NUMBER'
                else:
                    token = 'COLD NUMBER'
            else:
                token = re_punc.sub('',token)

            if token:
                return_tokens.append(token)

        return return_tokens

if __name__ == '__main__':
    import pandas as pd
    train_data = pd.read_csv(open('data/train.csv','r'),quotechar='"')
    tweets = train_data['tweet'].tolist()
    for tweet in tweets[:100]:
        print tweet
        print tokenize_tweet(tweet)

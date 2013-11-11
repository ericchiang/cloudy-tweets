#!/usr/bin/python

__author__    = "Eric Chiang"
__copyright__ = "Copyright 2013, Eric Chiang"
__email__     = "eric.chiang.m@gmail.com"

__license__   = "GPL"
__version__   = "3.0"

import csv
import re
import string

meta_tags = ['@mention','{link}']
alphanumeric_word = 'ALPHANUMERIC_WORD'
numeric_word = 'NUMERIC_WORD'
numbers = '0123456789'

re_number = re.compile(r'^\d*\.?\d+$')
re_temp   = re.compile(r'^\d+f$')
re_contains_num = re.compile('\d')
punct_no_apost = string.punctuation.replace("'",'')

ignore_words = ['the','to','in','a','and','is','or',
                'for','of','it','on']

pos_emoticons = [':)',':-)',': )',':D','=)']
neg_emoticons = [':(',':-(',': (','=(']
ntr_emoticons = [':/','=/',':\\','=\\',':S','=S',':|','=|']

"""
Parse CSV file into data matrix
"""
def parseTwitterCSV(tweet_csv):
    data = []
    tweet_reader = csv.reader(tweet_csv, delimiter=',', quotechar='"')
    for row in tweet_reader:
        data.append(row)
    return data

"""
Parse tweet into bag of words
"""
def parseTweet(tweet):
    bag_of_words = []

    if '!' in tweet:
        bag_of_words.append('!')

    if '?' in tweet:
        bag_of_words.append('?')
    
    for emoticon in pos_emoticons:
        if emoticon in tweet:
            bag_of_words.append(':)')
            break

    for emoticon in neg_emoticons:
        if emoticon in tweet:
            bag_of_words.append(':(')
            break

    for emoticon in ntr_emoticons:
        if emoticon in tweet:
            bag_of_words.append(':/')

    # Why not?
    if ';)' in tweet:
        bag_of_words.append(';)')

    for tag in meta_tags:
        if tag in tweet:
            bag_of_words.append(tag)
            tweet = tweet.replace(tag,' ')

    for punch_char in punct_no_apost:
        tweet = tweet.replace(punch_char,' ')

    tweet = tweet.split()
    for word in tweet:
        word = word.lower()
        if word in ignore_words:
            continue
        if word:
            if re_number.match(word):
                if float(word) > 100:
                    word = 'over 100'
                else:
                    word = 'over %s'  % (int(float(word) / 10.0) * 10,)
                bag_of_words.append('Numeric')
            elif re_contains_num.match(word):
                for number in numbers:
                    word = word.replace(number,'')
                    # Append a tag to the word to signify an alphanumeric value
                word = word + ' #num' 
                bag_of_words.append('Alphanumeric')
            bag_of_words.append(word)

    return bag_of_words

      

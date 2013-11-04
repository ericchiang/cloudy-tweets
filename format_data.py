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

re_number = re.compile(r'^\d*\.?\d+$')
re_temp   = re.compile(r'^\d+f$')
re_contains_num = re.compile('\d')
punct_no_apost = string.punctuation.replace("'",'')

ignore_words = ['the','to','in','a','and','is','or',
                'for','of','it','rt','on','be','so']

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
    for tag in meta_tags:
        tweet = tweet.replace(tag,' ')

    for punch_char in punct_no_apost:
        tweet = tweet.replace(punch_char,' ')

    tweet = tweet.split()
    for i in range(len(tweet)):
        word = tweet[i].lower()
        if word in ignore_words:
            continue
        if word:
            if re_number.match(word):
                if float(word) > 90.0:
                    word = 'N > 90'
                elif float(word) > 50.0:
                    word = '90 >= N > 50'
                elif float(word) < 10.0:
                    word = 'N < 10'
                else:
                    word = '50 >= N >= 10'

                #word = numeric_word
            elif re_contains_num.match(word):
                word = 'ALPHANUMERIC'
            bag_of_words.append(word)

    return bag_of_words

      

# -*- coding: utf-8 -*-

# Final Project
# CS175, Winter 2015
#
# Sentiment Analysis on Tweets 
#
# Authors:
#  Bing Hui Feng, 78912993, bhfeng@uci.edu 
#  Jessica Pham, 80859856, jessictp@uci.edu
#  Brian Chou, 72312823, bchou1@uci.edu

"""
    PROGRAM DESCRIPTION
        this script runs the preprocessing on the data set located at 
        http://help.sentiment140.com/for-students/ . Removes hash tags URLs
        and User tags
"""


#preprocessing
import pandas as pd
import numpy as np

wordcount = {}
numtweets = 0

def prepare_data(input_file:str) -> list:
    input_twitter_data = pd.read_csv(input_file, encoding='ISO-8859-1',error_bad_lines=False);
    twitter_data = input_twitter_data.as_matrix();
    tweets = twitter_data[:,5]; # numpy array
    tweet_sentiments = twitter_data[:,0].astype('str');
    return tweets, tweet_sentiments 
    
    
def replace_words(tweetText: str, user:bool, hashtag:bool, url: bool) -> None:
    replace_words = ['@', 'http', '//t.co', '.com', '.org', 'www', '.edu', '.net', '#']
    tokens = tweetText.split()
    for tok in reversed(tokens):
        for re in replace_words:
            if re in tok:
                if re == '@':
                    if tok.startswith(re) and len(tok) > 1 and user: 
                        tokens[tokens.index(tok)] = '<USER>'
                        break
                elif re == '#':
                    if tok.startswith(re) and len(tok) > 1 and hashtag:
                        tokens[tokens.index(tok)] = '<HASHTAG>'
                        break
                elif url:
                    tokens[tokens.index(tok)] = '<URL>'
                    break
    for t in tokens:
        if not t in wordcount.keys():
            wordcount[t] = 1
        else:
            wordcount[t] += 1
            
    return " ".join(tokens)
    

def preprocess_data(input_file:str, save_destination:str, user:bool, hashtag:bool, url: bool, neutral: bool):
    tweets, tweet_sentiments = prepare_data(input_file)   
    global numtweets     
    new_tweets = []
    new_sentiments = []
    
    for i in range(len(tweets)):
        if "2" in tweet_sentiments[i] and neutral:
            pass
        else:
            new_tweets.append(tweets[i])
            new_sentiments.append(tweet_sentiments[i])
        
    for i in range(len(new_tweets)):
        new_tweets[i] = replace_words(new_tweets[i], user, hashtag, url)
        numtweets += 1
            
    processed_tweet_data = np.column_stack((np.asarray(new_sentiments) , np.asarray(new_tweets)))
    np.savetxt( save_destination , processed_tweet_data, delimiter=",", fmt="\"%s\"")    

print("processing data from csv to data matrix....")

#preprocess_data('data/training.csv', "data/processed_train.csv", True, True, True)
#remove_neutral('data/testdata.csv', "data/processed_test_noneutral.csv")
#preprocess_data('data/testdata.csv', "processed_test2.csv", True, True, True)



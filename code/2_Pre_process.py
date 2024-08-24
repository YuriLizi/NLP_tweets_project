import os
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

import pandas as pd
import numpy as np
import sklearn
import contractions
import string
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def setup_nltk():
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('stopwords')

def lemmatize_word(text):
    """
        Lemmatize the tokenized words
    """
    lemmatizer = WordNetLemmatizer()
    lemma = [lemmatizer.lemmatize(word) for word in text]
    return lemma

def tokenize_text(df, column):
    df['tokenized'] = df[column].apply(word_tokenize)
    return df

def remove_stopwords(df, column):
    stop = set(stopwords.words('english'))
    df['stopwords_removed'] = df[column].apply(lambda x: [word for word in x if word not in stop])
    return df

def lemmatize_text(df, column):
    lemmatizer = WordNetLemmatizer()
    df['lemmatize_tweets'] = df[column].apply(lambda x: [lemmatizer.lemmatize(word) for word in x])
    df['lemmatize_tweets'] = df['lemmatize_tweets'].apply(lambda x: [word for word in x if word not in stopwords.words('english')])
    return df

def save_csv(df, filepath):
    df.to_csv(filepath, index=False)

if __name__ == "__main__":

    #setup_nltk()

    train_df = pd.read_csv("/home/cortica/2nd_degree/nlp/project/data/tweets_cnn_clean.csv")

    train_df = tokenize_text(train_df, 'tweet_clean')
    train_df = remove_stopwords(train_df, 'tokenized')
    train_df = lemmatize_text(train_df, 'stopwords_removed')

    print(train_df.head())
    save_csv(train_df, "/home/cortica/2nd_degree/nlp/project/data/testtweets_cnn_clean_lemmatized.csv")

import os
import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter("ignore")

import pandas as pd
import numpy as np
import sklearn
import contractions
# Packages for text pre-processing
import string
import re
import nltk

def print_versions():
    print("Python version:", sys.version)
    print("Version info.:", sys.version_info)
    print("numpy version:", np.__version__)
    print("pandas version:", pd.__version__)
    print("sklearn version:", sklearn.__version__)
    print("re version:", re.__version__)
    print("nltk version:", nltk.__version__)

def remove_URL(text):
    """
        Remove URLs
    """
    return re.sub(r"https?://\S+|www\.\S+", "", text)

def remove_html(text):
    """
        Remove  html
    """
    html = re.compile(r"<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});")
    return re.sub(html, "", text)

def remove_non_ascii(text):
    """
        Remove non-ASCII characters
    """
    return re.sub(r'[^\x00-\x7f]',r'', text)

def remove_special_characters(text):
    """
        remove special characters
    """
    emoji_pattern = re.compile(
        '['
        u'\U0001F600-\U0001F64F'  # emoticons
        u'\U0001F300-\U0001F5FF'  # symbols & pictographs
        u'\U0001F680-\U0001F6FF'  # transport & map symbols
        u'\U0001F1E0-\U0001F1FF'  # flags (iOS)
        u'\U00002702-\U000027B0'
        u'\U000024C2-\U0001F251'
        ']+',
        flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_punct(text):
    """
        Remove punctuation
    """
    return text.translate(str.maketrans('', '', string.punctuation))

def list_files(directory):
    for dirname, _, filenames in os.walk(directory):
        for filename in filenames:
            print(os.path.join(dirname, filename))

def read_csv(filepath):
    return pd.read_csv(filepath)

def clean_tweets(df):
    df["tweet_clean"] = df["tweet"].apply(lambda x: x.lower())
    df["tweet_clean"] = df["tweet_clean"].apply(lambda x: contractions.fix(x))
    df["tweet_clean"] = df["tweet_clean"].apply(lambda x: remove_URL(x))
    df["tweet_clean"] = df["tweet_clean"].apply(lambda x: remove_html(x))
    df["tweet_clean"] = df["tweet_clean"].apply(lambda x: remove_non_ascii(x))
    df["tweet_clean"] = df["tweet_clean"].apply(lambda x: remove_special_characters(x))
    df["tweet_clean"] = df["tweet_clean"].apply(lambda x: remove_punct(x))
    return df

def save_csv(df, filepath):
    df.to_csv(filepath, index=False)

if __name__ == "__main__":
    print_versions()
    list_files('/home/cortica/2nd_degree/nlp/project/data/tweets_cnn.csv')
    print("------------------------------------------------------------------")
    train_df = read_csv("/home/cortica/2nd_degree/nlp/project/data/tweets_cnn.csv")
    print(train_df.shape)
    print(train_df.head())

    print(train_df[train_df["date"] == "2021-06-05"]["tweet"].values[0])
    print(train_df[train_df["date"] == "2021-05-21"]["tweet"].values[0])

    train_df = clean_tweets(train_df)
    save_csv(train_df, "/home/cortica/2nd_degree/nlp/project/data/tweets_cnn_clean.csv")

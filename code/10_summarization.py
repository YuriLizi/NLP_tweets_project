import pandas as pd
from transformers import pipeline

# Load the summarization model
summarizer = pipeline(task="summarization",device=0)

def summarize_tweet(tweet, max_length=20, min_length=5):
    try:
        summary = summarizer(tweet, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
    except Exception as e:
        summary = f"Error: {str(e)}"
    return summary

# Load the CSV file containing tweets
input_csv = '/home/cortica/2nd_degree/nlp/project/data/tweets_cnn.csv'
#input_csv = '/home/cortica/2nd_degree/nlp/project/data/tweets_bbc_clean_short_test.csv'
tweets_df = pd.read_csv(input_csv)

# Ensure there is a 'tweet' column in the CSV
if 'tweet' not in tweets_df.columns:
    raise ValueError("Input CSV must contain a 'tweet' column")

# Summarize each tweet
tweets_df['summary'] = tweets_df['tweet'].apply(summarize_tweet)

# Save the summarized tweets to a new CSV file
output_csv = '/home/cortica/2nd_degree/nlp/project/data/outputs/charts/10_summarizatin/cnn_summarized_tweets.csv'
tweets_df.to_csv(output_csv, index=False)

print(f"Summarized tweets have been saved to {output_csv}")

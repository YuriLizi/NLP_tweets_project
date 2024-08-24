import pandas as pd
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download VADER lexicon
nltk.download('vader_lexicon')

# Load the CSV file
file_path = '/home/cortica/2nd_degree/nlp/project/data/tweets_cnn.csv'  # Replace with your file path
df = pd.read_csv(file_path)

# Initialize the sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Function to get sentiment score
def get_sentiment_score(tweet):
    sentiment = sid.polarity_scores(tweet)
    return sentiment['compound']

# Apply the sentiment analysis to each tweet
df['sentiment_score'] = df['tweet'].apply(get_sentiment_score)

# Convert the 'date' column to datetime
df['date'] = pd.to_datetime(df['date'])

# Group by date and calculate the mean sentiment score for each day
average_sentiment_per_day = df.groupby(df['date'].dt.date)['sentiment_score'].mean()

# Apply a rolling mean to smooth the graph
rolling_window_size = 50  # Adjust this value for more or less smoothing
smoothed_sentiment = average_sentiment_per_day.rolling(window=rolling_window_size).mean()

# Plot the smoothed average sentiment scores per day
plt.figure(figsize=(10, 6))
smoothed_sentiment.plot(kind='line')
plt.title('CNN Smoothed Average Sentiment Score of Tweets Per Day')
plt.xlabel('Date')
plt.ylabel('Average Sentiment Score')
plt.xticks(rotation=90)
plt.savefig("CNN Smoothed Average Sentiment Score of Tweets Per Day.png")

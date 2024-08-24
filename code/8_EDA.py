import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = '/home/cortica/2nd_degree/nlp/project/data/tweets_cnn.csv'  # Replace with your file path
df = pd.read_csv(file_path)

# Combine date and time into a single datetime column
df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])

# Extract date and hour information
df['date_only'] = df['datetime'].dt.date
df['hour'] = df['datetime'].dt.hour

# Histogram of number of tweets per day
plt.figure(figsize=(10, 6))
df['date_only'].value_counts().sort_index().plot(kind='bar')
plt.title('CNN Histogram of Number of Tweets per Day')
plt.xlabel('Date')
plt.ylabel('Number of Tweets')
plt.xticks(rotation=90)
plt.savefig("CNN Histogram of Number of Tweets per Day.png")

# Histogram of number of tweets per hour across all days
plt.figure(figsize=(10, 6))
df['hour'].value_counts().sort_index().plot(kind='bar')
plt.title('CNN Histogram of Number of Tweets per Hour')
plt.xlabel('Hour')
plt.ylabel('Number of Tweets')
plt.xticks(range(24))
plt.savefig("CNN Histogram of Number of Tweets per Hour.png")

# Trend of number of tweets per day
plt.figure(figsize=(10, 6))
df['date_only'].value_counts().sort_index().plot(kind='line')
plt.title('CNN Trend of Number of Tweets per Day')
plt.xlabel('Date')
plt.ylabel('Number of Tweets')
plt.xticks(rotation=90)
plt.savefig("CNN Trend of Number of Tweets per Day.png")

import pandas as pd
import spacy
import en_core_web_sm
# python -m spacy download en_core_web_sm
#nlp = en_core_web_sm.load()
# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Function to perform NER on a single tweet
def extract_entities(text):
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities

# Read the CSV file containing tweets
input_csv = '/home/cortica/2nd_degree/nlp/project/data/tweets_cnn_clean.csv'
tweets_df = pd.read_csv(input_csv)

# Perform NER on each tweet and store the results in a new column
tweets_df['Entities'] = tweets_df['tweet_clean'].apply(extract_entities)

# Save the results to a new CSV file
output_csv = 'cnn_tweets_clean_with_entities.csv'
tweets_df.to_csv(output_csv, index=False)

print(f"Processed {len(tweets_df)} tweets and saved results to {output_csv}")

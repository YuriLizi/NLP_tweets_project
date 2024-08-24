import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import sys
import os
from gensim.models import Word2Vec

def load_data(filepath):
    """
    Load the CSV file.
    """
    return pd.read_csv(filepath)

def train_word2vec(df, column):
    """
    Train a Word2Vec model on the lemmatized tweets.
    """
    sentences = df[column].apply(eval).tolist()  # Convert string representations of lists to actual lists
    model = Word2Vec(sentences, vector_size=300, window=5, min_count=1, workers=4)
    return model

def get_most_used_words(model, n=20):
    """
    Get the most used words using the Word2Vec model.
    """
    words_freq = [(word, model.wv.get_vecattr(word, 'count')) for word in model.wv.key_to_index]
    sorted_words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
    return sorted_words_freq[:n]

def generate_bar_chart(words_scores, output_file):
    """
    Generate a bar chart of the most used words by frequency and save it to a file.
    """
    words, frequencies = zip(*words_scores)
    plt.figure(figsize=(10, 6))
    plt.bar(words, frequencies)
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title('Most Used Words by Frequency')
    plt.xticks(rotation=45)
    plt.savefig(output_file)
    plt.close()

def generate_word_cloud(words_scores, output_file):
    """
    Generate a word cloud of the words based on frequencies and save it to a file.
    """
    word_cloud_dict = dict(words_scores)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_cloud_dict)
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(output_file)
    plt.close()

def get_output_filenames(filepath, save_location):
    """
    Generate output filenames based on the input file name and save location.
    """
    base_name = os.path.splitext(os.path.basename(filepath))[0]
    bar_chart_file = os.path.join(save_location, f"{base_name}_word2vec_bar_chart.png")
    word_cloud_file = os.path.join(save_location, f"{base_name}_word2vec_word_cloud.png")
    return bar_chart_file, word_cloud_file

if __name__ == "__main__":

    filepath = "/home/cortica/2nd_degree/nlp/project/data/tweets_cnn_clean_lemmatized.csv"
    save_location = "/home/cortica/2nd_degree/nlp/project/data/outputs/charts/word2vec"



    df = load_data(filepath)

    # Train Word2Vec model and find most used words
    word2vec_model = train_word2vec(df, 'lemmatize_tweets')
    most_used_words = get_most_used_words(word2vec_model)

    bar_chart_file, word_cloud_file = get_output_filenames(filepath, save_location)

    generate_bar_chart(most_used_words, bar_chart_file)
    generate_word_cloud(most_used_words, word_cloud_file)

    print(f"Bar chart saved as '{bar_chart_file}'")
    print(f"Word cloud saved as '{word_cloud_file}'")

    print("Most used words based on Word2Vec model:")
    for word, freq in most_used_words:
        print(f"{word}: {freq}")

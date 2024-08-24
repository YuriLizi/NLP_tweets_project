import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import sys
import os

def load_data(filepath):
    """
    Load the CSV file.
    """
    return pd.read_csv(filepath)

def compute_tfidf(df, column):
    """
    Compute TF-IDF values for the given column in the DataFrame.
    """
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df[column].apply(lambda x: ' '.join(eval(x))))
    feature_names = tfidf.get_feature_names_out()
    return tfidf_matrix, feature_names

def get_top_n_words(tfidf_matrix, feature_names, n=20):
    """
    Get the top N words by TF-IDF score.
    """
    sum_tfidf = tfidf_matrix.sum(axis=0)
    words_scores = [(word, sum_tfidf[0, idx]) for word, idx in zip(feature_names, range(sum_tfidf.shape[1]))]
    sorted_words_scores = sorted(words_scores, key=lambda x: x[1], reverse=True)
    return sorted_words_scores[:n]

def generate_bar_chart(words_scores, output_file):
    """
    Generate a bar chart of the top N words by TF-IDF score and save it to a file.
    """
    words, scores = zip(*words_scores)
    plt.figure(figsize=(10, 6))
    plt.bar(words, scores)
    plt.xlabel('Words')
    plt.ylabel('TF-IDF Score')
    plt.title('Top Words by TF-IDF Score')
    plt.xticks(rotation=45)
    plt.savefig(output_file)
    plt.close()

def generate_word_cloud(words_scores, output_file):
    """
    Generate a word cloud of the words based on TF-IDF scores and save it to a file.
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
    bar_chart_file = os.path.join(save_location, f"{base_name}_tfidf_bar_chart.png")
    word_cloud_file = os.path.join(save_location, f"{base_name}_tfidf_word_cloud.png")
    return bar_chart_file, word_cloud_file

if __name__ == "__main__":

    filepath = "/home/cortica/2nd_degree/nlp/project/data/tweets_cnn_clean_lemmatized.csv"
    save_location = "/home/cortica/2nd_degree/nlp/project/data/outputs/charts"


    df = load_data(filepath)

    tfidf_matrix, feature_names = compute_tfidf(df, 'lemmatize_tweets')
    top_words_scores = get_top_n_words(tfidf_matrix, feature_names)

    bar_chart_file, word_cloud_file = get_output_filenames(filepath, save_location)

    generate_bar_chart(top_words_scores, bar_chart_file)
    generate_word_cloud(top_words_scores, word_cloud_file)

    print(f"Bar chart saved as '{bar_chart_file}'")
    print(f"Word cloud saved as '{word_cloud_file}'")

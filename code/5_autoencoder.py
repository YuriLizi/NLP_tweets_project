import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, RepeatVector, TimeDistributed
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os

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
    bar_chart_file = os.path.join(save_location, f"{base_name}_autoencoder_bar_chart.png")
    word_cloud_file = os.path.join(save_location, f"{base_name}_autoencoder_word_cloud.png")
    return bar_chart_file, word_cloud_file


# Load the CSV file
input_file = '/home/cortica/2nd_degree/nlp/project/data/tweets_cnn_clean_lemmatized.csv'
save_location = "/home/cortica/2nd_degree/nlp/project/data/outputs/charts/autoencoder"
df = pd.read_csv(input_file)

# Assuming the text data is in a column named 'tweet'
texts = df['tweet_clean'].astype(str).tolist()

# Tokenize the text data
max_words = 10000  # Maximum number of words to consider
max_len = 40  # Maximum length of each tweet

tokenizer = Tokenizer(num_words=max_words, oov_token='<UNK>')
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
word_counts = tokenizer.word_counts
sorted_words = sorted(word_counts.items(), key=lambda item: item[1],reverse=True)

# Pad the sequences
data = pad_sequences(sequences, maxlen=max_len)

# Build the autoencoder model
embedding_dim = 128
latent_dim = 64  # Dimensionality of the latent space

input_text = Input(shape=(max_len,))
embedding = Embedding(input_dim=max_words, output_dim=embedding_dim, input_length=max_len)(input_text)
encoded = LSTM(latent_dim)(embedding)

decoded = RepeatVector(max_len)(encoded)
decoded = LSTM(embedding_dim, return_sequences=True)(decoded)
decoded = TimeDistributed(Dense(max_words, activation='softmax'))(decoded)

# Define the autoencoder, encoder, and decoder models
autoencoder = Model(input_text, decoded)
encoder = Model(input_text, encoded)

encoded_input = Input(shape=(latent_dim,))
decoded_layer = autoencoder.layers[-3](encoded_input)
decoded_layer = autoencoder.layers[-2](decoded_layer)
decoded_output = autoencoder.layers[-1](decoded_layer)
decoder = Model(encoded_input, decoded_output)

# Compile the autoencoder
autoencoder.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Prepare the data for training (LSTM expects the output to be one-hot encoded)
data_reshaped = np.expand_dims(data, -1)

# Train the autoencoder
autoencoder.fit(data, data_reshaped, epochs=1, batch_size=32, validation_split=0.2)
print("done training")
# Encode the text data
encoded_texts = encoder.predict(data)
print("done encoding")
# Decode the text data
#decoded_texts = decoder.predict(encoded_texts)

batch_size = 32
decoded_texts = []
for i in range(0, len(encoded_texts), batch_size):
    batch_encoded_texts = encoded_texts[i:i + batch_size]
    batch_decoded_texts = decoder.predict(batch_encoded_texts)
print("done decoding")
# Convert decoded sequences back to text
reverse_word_index = {v: k for k, v in word_index.items()}
decoded_texts_str = []
for decoded_sequence in batch_decoded_texts:
    decoded_str = ' '.join([reverse_word_index.get(np.argmax(word), '') for word in decoded_sequence])
    decoded_texts.append(decoded_str.strip())

# Save the encoded text data and decoded text data to a new CSV file
sorted_words = sorted_words[12:32]
encoded_df = pd.DataFrame(encoded_texts)
decoded_df = pd.DataFrame({'decoded_tweet': decoded_texts_str})
result_df = pd.concat([encoded_df, decoded_df], axis=1)
#result_df.to_csv('encoded_and_decoded_tweets.csv', index=False)


bar_chart_file, word_cloud_file = get_output_filenames(input_file, save_location)

generate_bar_chart(sorted_words, bar_chart_file)
generate_word_cloud(sorted_words, word_cloud_file)
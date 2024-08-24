import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


# Load the CSV file into a DataFrame
def load_data(file_path):
    return pd.read_csv(file_path)


# Preprocess the text data: tokenize and create sequences
def preprocess_text(df, max_vocab_size=10000, max_sequence_len=50):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=max_vocab_size)
    tokenizer.fit_on_texts(df['tweet'])  # Use the correct column name here
    sequences = tokenizer.texts_to_sequences(df['tweet'])
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_sequence_len, padding='post')
    return padded_sequences, tokenizer


# Split the data into training and validation sets
def split_data(padded_sequences, test_size=0.01):
    X_train, X_val = train_test_split(padded_sequences, test_size=test_size, random_state=42)
    return X_train, X_val


# Build an RNN model
def build_rnn_model(max_vocab_size, max_sequence_len, embedding_dim=64, rnn_units=128):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=max_vocab_size, output_dim=embedding_dim, input_length=max_sequence_len),
        tf.keras.layers.SimpleRNN(rnn_units, return_sequences=False),
        tf.keras.layers.Dense(max_vocab_size, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


# Train the model
def train_model(model, X_train, X_val, epochs=17, batch_size=128):
    y_train = np.roll(X_train, -1, axis=1)[:, -1]  # Shift and take the last word in the sequence
    y_val = np.roll(X_val, -1, axis=1)[:, -1]  # Same for validation set

    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size)


# Generate text using the trained model
def generate_text(model, tokenizer, seed_text, num_words_to_generate=50, max_sequence_len=50):
    for _ in range(num_words_to_generate):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = tf.keras.preprocessing.sequence.pad_sequences([token_list], maxlen=max_sequence_len, padding='pre')
        predicted = np.argmax(model.predict(token_list), axis=-1)

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break

        seed_text += " " + output_word

    return seed_text


# Save generated text to a file
def save_generated_text(text, file_path):
    with open(file_path, 'w') as f:
        f.write(text)


# Main function to run the text generation model and save the generated text
def main(file_path):
    df = load_data(file_path)
    padded_sequences, tokenizer = preprocess_text(df)

    X_train, X_val = split_data(padded_sequences)

    max_vocab_size = 10000
    max_sequence_len = padded_sequences.shape[1]

    model = build_rnn_model(max_vocab_size, max_sequence_len)
    train_model(model, X_train, X_val)

    seed_text = "the world will"  # Replace with a seed text to start the generation
    generated_text = generate_text(model, tokenizer, seed_text)

    output_file_path = '../generated_tweets.txt'  # File to save the generated text
    save_generated_text(generated_text, output_file_path)
    print(f"Generated text saved to {output_file_path}")


# Example usage
file_path = '/home/cortica/2nd_degree/nlp/project/data/tweets_bbc.csv'  # Replace with the path to your CSV file
main(file_path)

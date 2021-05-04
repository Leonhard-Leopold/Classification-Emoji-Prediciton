from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.python.keras import models, layers
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.metrics import accuracy_score
import re
import pandas
from keras.utils import to_categorical


def predict_emoji(tweet):
    x = re.compile(r'[^a-z\s]').sub(r'', re.compile(r'[\W]').sub(r' ', tweet.lower()))
    x = tokenizer.texts_to_sequences(np.array([x]))
    x = tf.keras.preprocessing.sequence.pad_sequences(x, maxlen=max_length)
    result = model.predict(x)
    print("'"+tweet+"' got the emoji: ",  np.argmax(result, axis=1)[0])


train_tweets = pandas.read_pickle("emoji_train.pkl")["tweet"]
train_emojis = pandas.read_pickle("emoji_train.pkl")["emoji_class"]
test_tweets = pandas.read_pickle("emoji_test.pkl")["tweet"]
test_emojis = pandas.read_pickle("emoji_test.pkl")["emoji_class"]
print(train_tweets.shape)
print(test_tweets.shape)

print(pandas.read_pickle("emoji_train.pkl").head())

print("Text preprocessing ...")
for i in range(len(train_tweets)):
    train_tweets[i] = re.compile(r'[^a-z0-9\s]').sub(r'', re.compile(r'[\W]').sub(r' ', train_tweets[i].lower()))
for i in range(len(test_tweets)):
    test_tweets[i] = re.compile(r'[^a-z0-9\s]').sub(r'', re.compile(r'[\W]').sub(r' ', test_tweets[i].lower()))

max_features = 20000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(train_tweets)
train_tweets = tokenizer.texts_to_sequences(train_tweets)
test_tweets = tokenizer.texts_to_sequences(test_tweets)

max_length = max(max(len(train_r) for train_r in train_tweets), max(len(train_r) for train_r in train_tweets))
train_tweets = tf.keras.preprocessing.sequence.pad_sequences(train_tweets, maxlen=max_length)
test_tweets = tf.keras.preprocessing.sequence.pad_sequences(test_tweets, maxlen=max_length)

print("Splitting dataset ...")
train_tweets, validation_tweets, train_emojis, validation_emojis = train_test_split(train_tweets, train_emojis, test_size=0.2)

input = layers.Input(shape=(max_length,))
x = layers.Embedding(max_features, 128)(input)
x = layers.LSTM(128, return_sequences=False)(x)
x = layers.Dropout(0.5)(x)
# x = layers.LSTM(128, return_sequences=False)(x)
# x = layers.Dropout(0.5)(x)
output = layers.Dense(7, activation="softmax")(x)

model = models.Model(inputs=input, outputs=output)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


try:
    model.load_weights('weights/emojis.hdf5')
    print("\nLoading previous model weights:\n")
except:
    print("\nNo weights found. Training new model\n")

print("Training Model:\n")
model.fit(train_tweets, to_categorical(train_emojis), batch_size=128, epochs=2, validation_data=(validation_tweets, to_categorical(validation_emojis)))

preds = model.predict(test_tweets)
print("\nPredictions:\n")
print(preds, "\n\n")

print('Accuracy score: {:0.4}'.format(accuracy_score(test_emojis, np.argmax(preds, axis=1))))

print("\nTesting model: ")

predict_emoji("I love this picture")
predict_emoji("Look at the sunset")
predict_emoji("This is so sad")
predict_emoji("Bolt won the race again! First place")

print("\n\nSaving model weights ...")
model.save_weights('weights/emojis.hdf5')

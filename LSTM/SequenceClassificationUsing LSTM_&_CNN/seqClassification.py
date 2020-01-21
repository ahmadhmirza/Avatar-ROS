# LSTM and CNN for sequence classification in the IMDB dataset
import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence


# fix random seed for reproducibility
numpy.random.seed(7)
# load the dataset but only keep the top n words, zero the rest
top_words = 20000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
# truncate and pad input sequences
max_review_length = 80
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
# create the model
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=15, batch_size=32)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


# =============================================================================
# from nltk import word_tokenize
# from keras.preprocessing import sequence
# word2index = imdb.get_word_index()
# test=[]
# for word in word_tokenize( "i love this movie"):
#      test.append(word2index[word])
# 
# test=sequence.pad_sequences([test],maxlen=max_review_length)
# prediction = model.predict(test)
# 
# print(prediction)
# =============================================================================

from keras.preprocessing import sequence
from keras.datasets import imdb
from keras.models import load_model
import numpy as np
import re

word_to_id = imdb.get_word_index()

testString = "it was not good"
#testString = "This movie was great"

strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
testString = testString.lower().replace("<br />", " ")
testString=re.sub(strip_special_chars, "", testString.lower())
print("Cleaned Data ", testString)

words = testString.split() #split string into a list
x_test = [[word_to_id[word] if (word in word_to_id and word_to_id[word]<=20000) else 0 for word in words]]
x_test = sequence.pad_sequences(x_test, maxlen=max_review_length) # Should be same which you used for training data
vector = np.array([x_test.flatten()])
print("Prediction is ",model.predict(vector),model.predict_classes(vector))


testString = "This movie was great"

strip_special_chars = re.compile("[^A-Za-z0-9 ]+")
testString = testString.lower().replace("<br />", " ")
testString=re.sub(strip_special_chars, "", testString.lower())
print("Cleaned Data ", testString)

words = testString.split() #split string into a list
x_test = [[word_to_id[word] if (word in word_to_id and word_to_id[word]<=20000) else 0 for word in words]]
x_test = sequence.pad_sequences(x_test, maxlen=max_review_length) # Should be same which you used for training data
vector = np.array([x_test.flatten()])
print("Prediction is ",model.predict(vector),model.predict_classes(vector))
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 10:13:53 2020

@author: ahmad
"""

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import cv2
from collections import deque
"""
Section-1 : Data Preperation
"""
class_Mappings= {
        "w01" : [1,"Start"],
        "w02" : [2,"Stop"],
        "w03" : [3,"Hello"],
        "p01" : [4,"How are you?"],
        "p02" : [5,"Nice to meet you?"],
        "p03" : [6,"Thank you!"],
        "c00" : [0,"Unclassified"]
        }
p01_path = r"/home/ahmad/Desktop/Project_Avatar_LipReading_DataSet/p01/"
p02_path = r"/home/ahmad/Desktop/Project_Avatar_LipReading_DataSet/p02/"

# Number of folders to iterate in the training directories
p01_dataEntries = 45
p02_dataEntries = 35
CLASS_MAPPING_CLASS_IND = 0

X_data=[]  #Training Data
Y_data=[]  #Corresponding Classes

# read data for P01
from skimage.transform import resize 
X=[]
for folder in range(1,p01_dataEntries+1):
    dataPath = os.path.join(p01_path,(str(folder)+"/*.*"))
    batchList = []
    try:
        for file in glob.glob(dataPath):
            img= cv2.imread(file,cv2.IMREAD_GRAYSCALE)
            img_resized = resize(img, preserve_range=True, output_shape=(224,224)).astype(int)
            #img = plt.imread(file)
            height,width = img_resized.shape
            #print(height,width)
            batchList.append(img_resized)
    except:
        print("exception")
    
    size = 224,224
    placeHolderMatrix = np.zeros(size, dtype=np.uint8)
    #placeHolderImage = cv2.fromarray(placeHolderMatrix)
    if len(batchList)<50:
        for i in range(0,(50-len(batchList))):
            batchList.append(placeHolderMatrix)
#    flat_list = []
#    for sublist in batchList:
#        for item in sublist:
#            flat_list.append(item)
    #X = np.array(batchList)    # converting list to array
    X.append(batchList)
    Y_data.append(class_Mappings["p01"][CLASS_MAPPING_CLASS_IND])
X_data=np.array(X)
X_data_flat=X_data.flatten()
##############################################################################    

################################################################################
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
MAX_SEQUENCE_LENGTH = 300
from keras.utils import np_utils
dummy_y = np_utils.to_categorical(Y_data) 

X_train, X_test, Y_train, Y_test = train_test_split(X_data,dummy_y, test_size = 0.10, random_state = 42)
print(X_train.shape,Y_train.shape)
print(X_test.shape,Y_test.shape)

X_train=X_train.flatten()
#X_train = X_train.reshape(40, 70*224*224)      # converting to 1-D
#X_test = X_test.reshape(5, 70*224*224)
#
print(X_train.shape)
#print(X_test.shape,Y_test.shape)

# The maximum number of words to be used. (most frequent)
MAX_NB_WORDS = 10000
# This is fixed.
EMBEDDING_DIM = 100

#################################
model = Sequential()

#model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X_train.shape[0]))
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=1))

model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
#model.add(Dense(units=1024, activation=tf.nn.relu,input_shape=X_train.shape[1]))
model.add(Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

epochs = 5
batch_size = 1

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,validation_split=0.1,callbacks=[EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

print("Saving the Model")
#from keras.models import model_from_json

# serialize model to JSON
model_json = model.to_json()
with open("TextClassificationModel.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("TextClassificationModel.h5")
print("Saved model to disk")

##################################
accr = model.evaluate(X_test,Y_test)
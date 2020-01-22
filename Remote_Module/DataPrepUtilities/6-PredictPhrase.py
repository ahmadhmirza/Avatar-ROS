#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 15:33:15 2019

@author: ahmad
"""
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np    # for mathematical operations
#import tensorflow.compat.v1 as tf
import tensorflow as tf
tf.disable_v2_behavior()
from keras.utils import np_utils
from skimage.transform import resize 
from keras.applications.vgg16 import preprocess_input
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
#from keras.layers import Dense, InputLayer, Dropout
from keras.layers import Dense, InputLayer
from keras.models import model_from_json

from pypac import pac_context_for_url
with pac_context_for_url("https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

#Loading VGG16 model and saving it as base_model

print("VGG Model loaded...") 

modelJson = r'/home/ahmad/Avatar/MachineLearning/data_prep_v1/TrainedModel/model.json'
modelH5 = r'/home/ahmad/Avatar/MachineLearning/data_prep_v1/TrainedModel/model.h5'


NumToLang = {
        0:"Un-Classified",
        1:"Start",
        2:"Stop",
        3:"Hello",
        4:"How are you?",
        5:"Nice to Meet you",
        6:"Thank you"
        }


TestImage_list=[
    r'/home/ahmad/Avatar/MachineLearning/data_prep_v1/Arranged_training_data/2_Test_Data/p01/1.jpg',
    r'/home/ahmad/Avatar/MachineLearning/data_prep_v1/Arranged_training_data/2_Test_Data/p01/2.jpg',
    r'/home/ahmad/Avatar/MachineLearning/data_prep_v1/Arranged_training_data/2_Test_Data/p01/3.jpg',
    r'/home/ahmad/Avatar/MachineLearning/data_prep_v1/Arranged_training_data/2_Test_Data/p01/4.jpg',
    r'/home/ahmad/Avatar/MachineLearning/data_prep_v1/Arranged_training_data/2_Test_Data/p01/5.jpg',
    r'/home/ahmad/Avatar/MachineLearning/data_prep_v1/Arranged_training_data/2_Test_Data/p02/1.jpg',
    r'/home/ahmad/Avatar/MachineLearning/data_prep_v1/Arranged_training_data/2_Test_Data/p02/2.jpg',
    r'/home/ahmad/Avatar/MachineLearning/data_prep_v1/Arranged_training_data/2_Test_Data/p02/3.jpg',
    r'/home/ahmad/Avatar/MachineLearning/data_prep_v1/Arranged_training_data/2_Test_Data/p02/4.jpg',
    r'/home/ahmad/Avatar/MachineLearning/data_prep_v1/Arranged_training_data/2_Test_Data/p02/5.jpg',
    r'/home/ahmad/Avatar/MachineLearning/data_prep_v1/Arranged_training_data/2_Test_Data/p03/1.jpg',
    r'/home/ahmad/Avatar/MachineLearning/data_prep_v1/Arranged_training_data/2_Test_Data/p03/2.jpg',
    r'/home/ahmad/Avatar/MachineLearning/data_prep_v1/Arranged_training_data/2_Test_Data/p03/3.jpg',
    r'/home/ahmad/Avatar/MachineLearning/data_prep_v1/Arranged_training_data/2_Test_Data/p03/4.jpg',
    r'/home/ahmad/Avatar/MachineLearning/data_prep_v1/Arranged_training_data/2_Test_Data/p03/5.jpg']


# load json and create model
json_file = open(modelJson, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(modelH5)
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

print("Pre-trained model loaded successfully")


print("***** Ready for testing *****")

count = 0
for images in TestImage_list:
    test_image = []
    img = plt.imread(images)
    test_image.append(img)
    test_img = np.array(test_image)
    count+=1
    
    test_image = []
    for i in range(0,test_img.shape[0]):
         a = resize(test_img[i], preserve_range=True, output_shape=(224,224)).astype(int)
         test_image.append(a)
    test_image = np.array(test_image)
    
    # preprocessing the images
    test_image = preprocess_input(test_image, mode='tf')
     
    # extracting features from the images using pretrained model
    test_image = base_model.predict(test_image)
    
    # converting the images to 1-D form
    test_image = test_image.reshape(1, 7*7*512)
    
    # zero centered images
    test_image = test_image/test_image.max()
    #
    prediction = loaded_model.predict_classes(test_image)
    print(prediction)
    
    print("Prediction: Person " + str(count) + " said : " + NumToLang.get(prediction[0]))


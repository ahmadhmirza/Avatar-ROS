#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
@author: Ahmad Hassan Mirza - ahmadhassan.mirza@gmail.com
------------------------------------------------------------------------------
Script for training and validation of CNN model based on VGG16 pre-trained 
model
------------------------------------------------------------------------------

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import numpy as np 
import matplotlib.pyplot as plt
import os
import tensorflow as tf
tf.disable_v2_behavior()
from keras.layers import Dropout

class_Mappings= {
        "w01" : [1,"Start"],
        "w02" : [2,"Stop"],
        "w03" : [3,"Hello"],
        "p01" : [4,"How are you?"],
        "p02" : [5,"Nice to meet you?"],
        "p03" : [6,"Thank you!"],
        "c00" : [0,"Unclassified"]
        }


p01_path = r"/home/ahmad/catkinJava_ws/Remote_Module/DataPrepUtilities/Project_Avatar_LipReading_DataSet/SingleData/p01/"
p02_path = r"/home/ahmad/catkinJava_ws/Remote_Module/DataPrepUtilities/Project_Avatar_LipReading_DataSet/SingleData/p02/"
p03_path = r"/home/ahmad/catkinJava_ws/Remote_Module/DataPrepUtilities/Project_Avatar_LipReading_DataSet/SingleData/p03/"

dataEntries = 0

Data_Info = {
        "P01": [90,p01_path],
        "P02": [90,p02_path],
        "P03": [70,p03_path]
        }
CLASS_MAPPING_CLASS_IND = 0

y_data=[]  #Corresponding Classes

X = [ ]


#Read images from disk
for item in  Data_Info:
    dataEntries = Data_Info[item][0]   
    dataPath = Data_Info[item][1] 
    count=1
    for count in range(1,dataEntries+1):
        try:
            image_name = str(count)+".jpg"
            img = plt.imread(os.path.join(dataPath + image_name))
            #print(os.path.join(basePath + img_name))
            X.append(img)  # storing each image in array X
            y_data.append(class_Mappings["p01"][CLASS_MAPPING_CLASS_IND])
            count += 1
        except:
            print("exception")
        
X = np.array(X) 
print ("Total training images for training dataset: " + str(len(X)))


# =============================================================================
# Preparing:
# Training images, and
# Their corresponding class
# =============================================================================
from keras.utils import np_utils
OHC_y = np_utils.to_categorical(y_data)    # one hot-encoding Classes

# =============================================================================
# Using VGG16 pretrained model which takes an input image of shape 
# (224 X 224 X 3). Since the images might be in a different size, a reshaping
# step is also performed here to the specified size
# =============================================================================
from skimage.transform import resize 
img = []
print ("Resizing Images...")
total = len(X)
current = 0
for i in range(0,X.shape[0]):
    current +=1
    # reshaping to 224*224*3
    img_temp = resize(X[i], preserve_range=True, output_shape=(224,224)).astype(int)      
    img.append(img_temp)
X = np.array(img)
print ("Images Resized...")

# =============================================================================
#  Pre-Processing input data as per VGG16's specs
# =============================================================================
print ("Pre-processing Images...")
from keras.applications.vgg16 import preprocess_input
X = preprocess_input(X, mode='tf')
print("Pre-Processing Done...")

# Dividing the data set into train and test data sets
print("Preparing train and test data..") 
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, OHC_y, test_size=0.5, random_state=42)
print("Data prepared...")


# =============================================================================
#  Building the model
# =============================================================================
import keras
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, InputLayer
print(keras.__version__)
from pypac import pac_context_for_url
with pac_context_for_url("https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

#Loading VGG16 model and saving it as base_model
print("VGG Model loaded...")   

############################################################################### 
# include_top=False to remove the top layer
# top layer of the model will be re.defined to fit the data set

# Here features from the dataset are extracted using the VGG model and
# These extracted features are then used to train the model further.
###############################################################################

print("Making predictions...")
X_train = base_model.predict(X_train,batch_size = 20)
X_valid = base_model.predict(X_valid,batch_size = 20)
X_train.shape, X_valid.shape

print(X_train.shape)
print(X_valid.shape)
# Reshaping input data to 1-D as CNN only accepts data with this dimension
print("Converting dataset to 1-D")
X_train = X_train.reshape(X_train.shape[0], 7*7*512)
X_valid = X_valid.reshape(X_valid.shape[0], 7*7*512)
print("1-D Conversion : Done")

###############################################################################
#print("preprocessing images to zero center...")
##preprocess the images and zero-centering -> helps the model converge faster
#train = X_train/X_train.max()      # centering the data
#X_valid = X_valid/X_train.max()
#print("Zero Centering: Done")
##############################################################################
#Build the model in 3steps:
# 1.Building the model
# 2.Compiling the model
# 3.Training the model

print("Building the model-Stage1")

EPOCHS = 5
# =============================Stage -1 =======================================
model = Sequential()
model.add(InputLayer((7*7*512,)))    # input layer
model.add(Dense(units=500, activation=tf.nn.relu,input_shape=X_train.shape)) # hidden layer
model.add(Dropout(0.1))
model.add(Dense(5,input_shape=(500,), activation='softmax'))    # output layer

model.summary()
print("Stage-1: Done")
###############################################################################
# =============================Stage -2 =======================================
print("===Stage2===")
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print("Stage-2: Done")
#############################################################################
# =============================Stage -3 =======================================
print("===Stage3===")
#epoch 87/97 - gave max val_accuracy
history = model.fit(X_train, y_train, epochs=EPOCHS, validation_data=(X_valid, y_valid))
print("Stage-3: Done")
print("Building the Model : Done")

#evaluate the model

scores = model.evaluate(X_train,y_train)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(scores[0],scores[1]))


plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
plt.show()
plt.savefig('/home/ahmad/catkinJava_ws/Remote_Module/DataPrepUtilities/Model/Model_Loss_Plot.png')

print("Saving the Model")

modelJson = r'/home/ahmad/catkinJava_ws/Remote_Module/DataPrepUtilities/Model/model.json'
modelH5 = r'/home/ahmad/catkinJava_ws/Remote_Module/DataPrepUtilities/Model/model.h5'
# serialize model to JSON
model_json = model.to_json()
with open(modelJson, "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(modelH5)
print("Saved model to disk")
#model.save(TF_Model)

print("============== END OF SCRIPT ==============")
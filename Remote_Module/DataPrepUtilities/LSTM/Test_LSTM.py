#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Created on Fri Jan 31 16:51:14 2020
@author: Ahmad Hassan Mirza - ahmadhassan.mirza@gmail.com
------------------------------------------------------------------------------
Script To test the trained LSTM model on data on disk
------------------------------------------------------------------------------

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
# numpy and scipy
import numpy as np
# OpenCV
import cv2
###############################################################################
import os
#import tensorflow.compat.v1 as tf
import tensorflow as tf
tf.disable_v2_behavior()
from keras.models import model_from_json
###############################################################################
from os import walk
from natsort import natsorted, ns
##############################################################################

###########CONSTANTS#################
MAX_IMAGES_IN_SEQUENCE = 42  
IMAGESIZE=[150,150]

SAMPLES     = 1
TIME_STEP   = MAX_IMAGES_IN_SEQUENCE
FEATURES    = IMAGESIZE[0]*IMAGESIZE[1]

Data_Info = {
        "p01": [50,"p01",1],
        "p02": [40,"p02",2],
        "p03": [40,"p03",3],
        }


modelJson = r'/home/ahmad/Avatar/MachineLearning/LSTM_Models/vLSTM/1/model.json'
modelH5 = r'/home/ahmad/Avatar/MachineLearning/LSTM_Models/vLSTM/1/model.h5'
NumToLang = {
0:"Un-Classified",
1:"How are you?",
2:"Nice to Meet you",
3:"Thank you"
}

#########################################################
def readImagesInPath(path):
    fileList=[]
    img_Names_List=[]
    images=[]
    for (dirpath, dirnames, filenames) in walk(path):
        #Read all the images in the current directory
        #arrange them in the list by name so the sequence in maintained
        fileList.extend(filenames)
        fileList=natsorted(fileList, alg=ns.IGNORECASE)
        #Read the image in the openCV array
        for element in fileList:
            if ".jpg" in element:
                img_Names_List.append(element)
        for image in img_Names_List:
            image_path = os.path.join(path,image)
            img = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
            #plt.show(img)
            if img is not None:
                img=cv2.resize(img,(IMAGESIZE[0],IMAGESIZE[1]))
                images.append(img)
        return images

def reshapeArray_1(inArray,mSamples,mTimeStep,mFeatures):
    x= inArray.reshape(mSamples,mTimeStep,mFeatures)
    return x


def make_zeros(n_rows: int, n_columns: int):
    matrix = []
    for i in range(n_rows):
        matrix.append([0] * n_columns)
    return matrix

def addPadding(threshold,ImageSequence):
    length = len(ImageSequence)
    count = abs(threshold -length)
    PaddingMatrix = make_zeros(IMAGESIZE[0],IMAGESIZE[1])
    
    pendulum = 0
    
    if(length<threshold):
        print(str(count) + " Images will be added as padding.")
        for x in range(0,count):
            if pendulum == 0:
                ImageSequence.insert(0,PaddingMatrix)
                pendulum = 1
            elif pendulum == 1:
                ImageSequence.append(PaddingMatrix)
                pendulum = 0
    if(length>threshold):
        print(str(count) + " Images will be removed and not considered.")
        for x in range(0,count):
            if pendulum == 0:
                del ImageSequence[0]
                pendulum = 1
            elif pendulum == 1:
                ind = len(ImageSequence)-1
                del ImageSequence[ind]
                pendulum = 0
    return ImageSequence

##################MAIN #######################


##### STAGE -1 READ IMAGES ###########
    
samplesList = []
samplePath = r"/home/ahmad/Desktop/TestData/"
sample = readImagesInPath(samplePath)
sample_padded=addPadding(MAX_IMAGES_IN_SEQUENCE,sample)
#convert list to array:
sample_array=np.asarray(sample_padded)
   
x_data=reshapeArray_1(sample_array,SAMPLES,TIME_STEP,FEATURES)
 
########## PREDICTION STAGE #######

json_file = open(modelJson, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights(modelH5)
print("Loaded model from disk")         
# evaluate loaded model on test data
loaded_model.compile(loss='mean_squared_error', optimizer='adam',metrics=['accuracy'])       
print("Pre-trained model loaded successfully")
#######################################################################

prediction = loaded_model.predict_classes(x_data)
print("######################")
print(prediction)
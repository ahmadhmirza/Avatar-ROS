#!/usr/bin/env python
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
------------------------------------------------------------------------------
Script to make prediction once the transmission from Remote device is complete
------------------------------------------------------------------------------
Created on Sun Nov 10 10:10:10 2019
@author: Ahmad Hassan Mirza - ahmadhassan.mirza@gmail.com
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
_version__=  '0.1'


################################ ROS- imports #################################
import sys, time
# numpy and scipy
import numpy as np
from scipy.ndimage import filters
# OpenCV
import cv2
# Ros libraries
import roslib
import rospy
from std_msgs.msg import Int32

###############################################################################
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np    # for mathematical operations
#import tensorflow.compat.v1 as tf
import tensorflow as tf
tf.disable_v2_behavior()
import keras
from keras.utils import np_utils
from skimage.transform import resize 
from keras.applications.vgg16 import preprocess_input
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
#from keras.layers import Dense, InputLayer, Dropout
from keras.layers import Dense, InputLayer
from keras.models import model_from_json

class LipReading_Predictor:
    def __init(self):#
##################### Part -1 : Load pre-trained model ########################
        from pypac import pac_context_for_url
        with pac_context_for_url("https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"):
            base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))        
        #Loading VGG16 model and saving it as base_model        
        print("VGG Model loaded...") 
        
############## Part -1.1 : Load Model Trained for lip-reading #################
        modelJson = r'/home/ahmad/Avatar/MachineLearning/data_prep_v1/TrainedModel/model.json'
        modelH5 = r'/home/ahmad/Avatar/MachineLearning/data_prep_v1/TrainedModel/model.h5'        
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

    def getClassToPhraseMap(int classNumber):
        NumToLang = {
        0:"Un-Classified",
        1:"Start",
        2:"Stop",
        3:"Hello",
        4:"How are you?",
        5:"Nice to Meet you",
        6:"Thank you"
        }
        return NumToLang.get(prediction[classNumber])
        
    def performPrediction(img):
        test_image = []
        test_image.append(img)
        test_img = np.array(test_image)
        
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
        print("Prediction Class  : " + prediction)
        
        print("Prediction Phrase : " + getClassToPhraseMap(prediction[0]))
        print("==========================================================")

VERBOSE=False
class Ros_PredictionFlag:
        
    def __init__(self):
        '''Initialize ros subscriber'''
        global predictionFlag
        global prev_predictionFlag 
        predictor = LipReading_Predictor()
        
        predictionFlag = 1
        prev_predictionFlag = predictionFlag
        # subscribed Topic
        self.subscriber = rospy.Subscriber("/prediction_flag",
            Int32, self.callback,  queue_size = 1)
        
        if VERBOSE :
            print ("subscribed to /prediction_flag")


    def callback(self, ros_data):
        
        if VERBOSE :
            print ('Received Prediction_Flag: "%s"' % ros_data)
        predictionFlag = ros_data.data       
        if (prev_predictionFlag==0) & (predictionFlag==0) :
            print("Receiving transmission from remote client...")
            prev_predictionFlag = predictionFlag
            
        if (prev_predictionFlag==0) & (predictionFlag==1) :
            print("Data recieved, analysing and performing prediction...")
            predictor.performPrediction()
            prev_predictionFlag = predictionFlag
            
        if (prev_predictionFlag==1) & (predictionFlag==0) :
            print("Standing by for next transmission...")
            predictor.performPrediction(img)
            prev_predictionFlag = predictionFlag
            
        if (prev_predictionFlag==1) & (predictionFlag==1) :
            prev_predictionFlag = predictionFlag
            pass
            
##############################################################################

def main(args):
    '''Initializes and cleanup ros node'''
    ic = Ros_PredictionFlag()
    rospy.init_node('Avatar_Desktop', anonymous=False)
    print("ok")
    try:
        rospy.spin()
        
    except KeyboardInterrupt:
        print ("Shutting down ROS Image feature detector module")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
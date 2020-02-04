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
import sys
# numpy and scipy
import numpy as np
# OpenCV
import cv2
# Ros libraries
import rospy
from std_msgs.msg import Int32

###############################################################################
import os
import matplotlib.pyplot as plt
#import tensorflow.compat.v1 as tf
import tensorflow as tf
tf.disable_v2_behavior()
from keras.models import model_from_json
###############################################################################
from os import walk
import shutil
from natsort import natsorted, ns
##################
import dlib
###############################################################################

class LipReading_Predictor:
    def __init(self):#
        print("Prediction class initialized successfully")

    def getClassToPhraseMap(classNumber):
        NumToLang = {
        0:"Un-Classified",
        1:"How are you?",
        2:"Nice to Meet you",
        3:"Thank you"
        }
        return NumToLang.get(classNumber)
        
    def performPrediction(x_data):
############## Part -1.1 : Load Model Trained for lip-reading #################
        
## vLSTM--1
#        modelJson = r'/home/ahmad/Avatar/MachineLearning/LSTM_Models/vLSTM/1/model.json'
#        modelH5 = r'/home/ahmad/Avatar/MachineLearning/LSTM_Models/vLSTM/1/model.h5'
#vLSTM--2
        modelJson = r'/home/ahmad/Avatar/MachineLearning/LSTM_Models/vLSTM/2/model.json'
        modelH5 = r'/home/ahmad/Avatar/MachineLearning/LSTM_Models/vLSTM/2/model.h5'  
# sLSTM--1
        #modelJson = r'/home/ahmad/Avatar/MachineLearning/LSTM_Models/sLSTM/model.json'
        #modelH5 = r'/home/ahmad/Avatar/MachineLearning/LSTM_Models/sLSTM/model.h5'          
        
        
        # load json and create model
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
        return prediction[0]
    
        

VERBOSE=False
class Ros_PredictionFlag: 
    def __init__(self):
        
        '''Initialize ros subscriber'''
        #self.predictor = LipReading_Predictor()
        
        self.predictionFlag = 1
        self.prev_predictionFlag = 1
        # subscribed Topic
        self.subscriber = rospy.Subscriber("/prediction_flag",
            Int32, self.callback,  queue_size = 1)
        
        if VERBOSE :
            print ("subscribed to /prediction_flag")
########################################################

    def displayImage(self,img):
        plt.imshow(img)
        plt.show()
        
    def make_zeros(self,n_rows: int, n_columns: int):
        matrix = []
        for i in range(n_rows):
            matrix.append([0] * n_columns)
        return matrix        
        
    def reshapeArray_1(self,inArray,mSamples,mTimeStep,mFeatures):
        x= inArray.reshape(mSamples,mTimeStep,mFeatures)
        return x  
    
    def addPadding(self,threshold,ImageSequence):
        IMAGESIZE=[150,150]
        length = len(ImageSequence)
        count = abs(threshold -length)
        PaddingMatrix = self.make_zeros(IMAGESIZE[0],IMAGESIZE[1])
       
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

    def prepareData(self):
        MAX_IMAGES_IN_SEQUENCE = 42  
        IMAGESIZE=[150,150]
        SAMPLES     = 1
        TIME_STEP   = MAX_IMAGES_IN_SEQUENCE
        FEATURES    = IMAGESIZE[0]*IMAGESIZE[1]
        
        outputDir = 'Pred_Data'
        predPath = r'/home/ahmad/Avatar/Prediction_Path'
        dlibPredPath = r'/home/ahmad/Avatar/MachineLearning/data_prep_v1/shape_predictor_68_face_landmarks.dat'
        
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(dlibPredPath)
        if os.path.exists(os.path.join(predPath,outputDir)):
                print("output dir already exists. Deleting the existing folder...")
                shutil.rmtree(os.path.join(predPath,outputDir))
                
        os.mkdir(os.path.join(predPath,outputDir))

#### READ INCOMMING IMAGES FROM THE PATH SPECIFIED       
        fileList=[]
        img_Names_List=[]
        images=[]
        for (dirpath, dirnames, filenames) in walk(predPath):
            #Read all the images in the current directory
            #arrange them in the list by name so the sequence in maintained
            fileList.extend(filenames)
            fileList=natsorted(fileList, alg=ns.IGNORECASE)
            #Read the image in the openCV array
            for element in fileList:
                if ".jpg" in element:
                    img_Names_List.append(element)
            for image in img_Names_List:
                image_path = os.path.join(predPath,image)
                img = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
                #plt.show(img)
                if img is not None:
                    img=cv2.resize(img,(IMAGESIZE[0],IMAGESIZE[1]))
                    images.append(img)
                    
        print(str(len(images)) + " frames recieved...")
# INCOMMING IMAGES CONTAIN FACE-ROI
# NEXT SECTION CROPS THEM TO LIP-ROI        
        print("Extracting Lips ROI...")
        lipsROI_list=[]
        for img in images:
            faces = detector(img)
            #print(faces)
            for face in faces:
                x1 = face.left()
                y1 = face.top()
                x2 = face.right()
                y2 = face.bottom()
                #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                landmarks = predictor(img, face)
            try:    
                x1 = landmarks.part(6).x  
                y1 = landmarks.part(2).y
        
                x2 = landmarks.part(10).x
                y2 = landmarks.part(6).y
        
                h=y2-y1
                w=x2-x1
    
                crop_img = img[y1:y1+h, x1:x1+w]
    
                crop_img = cv2.resize(crop_img,(150,150))
                lipsROI_list.append(crop_img)
            except:
                print("ERROR")
        #Delete processed files
        for file in fileList:
            try:
                os.remove(os.path.join(predPath,file))
            except:
                pass

# At this point lipsROI_list contains all the lip images in sequence.
        sample_padded=self.addPadding(MAX_IMAGES_IN_SEQUENCE,lipsROI_list)
        sample_array=np.asarray(sample_padded)
        TrainingData_FINAL=self.reshapeArray_1(sample_array,SAMPLES,TIME_STEP,FEATURES)   
        
        return TrainingData_FINAL
    
    def callback(self, ros_data):
        
        
        if VERBOSE :
            print ('Received Prediction_Flag: "%s"' % ros_data)
        
        self.predictionFlag = ros_data.data       
        if (self.prev_predictionFlag==0) & (self.predictionFlag==0) :
            print("Receiving transmission from remote client...")
            self.prev_predictionFlag = self.predictionFlag
            
        if (self.prev_predictionFlag==0) & (self.predictionFlag==1) :
            print("Data recieved, analysing and performing prediction...")
            images = self.prepareData()
            predictionClass = LipReading_Predictor.performPrediction(images)
            print("Prediction Value: " + str(predictionClass))
            print("Prediction Phrase : " + str(LipReading_Predictor.getClassToPhraseMap(predictionClass))) 
            print("==========================================================")
            self.prev_predictionFlag = self.predictionFlag
            
        if (self.prev_predictionFlag==1) & (self.predictionFlag==0) :
            print("Standing by for next transmission...")

            self.prev_predictionFlag = self.predictionFlag
            
        if (self.prev_predictionFlag==1) & (self.predictionFlag==1) :
            self.prev_predictionFlag = self.predictionFlag
            pass
            
##############################################################################

def main(args):
    '''Initializes and cleans up ros node'''
    ic = Ros_PredictionFlag()
    rospy.init_node('Avatar_Desktop_Prediction', anonymous=False)
    print("ok")
    try:
        rospy.spin()
        
    except KeyboardInterrupt:
        print ("Shutting down ROS Image feature detector module")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)
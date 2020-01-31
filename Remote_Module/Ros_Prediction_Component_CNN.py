#!/usr/bin/env python
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
------------------------------------------------------------------------------
Script to make prediction once the transmission from Remote device is complete
FOR CNN MODEL
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
import tensorflow as tf
tf.disable_v2_behavior()
from skimage.transform import resize 
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16
from keras.models import model_from_json
###############################################################################
from os import walk
import shutil
from natsort import natsorted, ns
##################
import dlib
from pypac import pac_context_for_url
###############################################################################

class LipReading_Predictor:
    def __init(self):#
        print("Prediction class initialized successfully")
##################### Part -1 : Load pre-trained model ########################
#        self.image_path = r'/home/ahmad/Avatar/Prediction_Path'
#        
#        with pac_context_for_url("https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"):
#            self.base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))        
#        #Loading VGG16 model and saving it as base_model        
#        print("VGG Model loaded...") 
        
############## Part -1.1 : Load Model Trained for lip-reading #################


    def getClassToPhraseMap(classNumber):
        NumToLang = {
        0:"Un-Classified",
        1:"Start",
        2:"Stop",
        3:"Hello",
        4:"How are you?",
        5:"Nice to Meet you",
        6:"Thank you"
        }
        return NumToLang.get(classNumber)
        
    def performPrediction(img):
        with pac_context_for_url("https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"):
            base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3)) 
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
        #######################################################################
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
        
    def prepareData(self):
        outputDir = 'Pred_Data'
        predPath = r'/home/ahmad/Avatar/Prediction_Path'
        dlibPredPath = r'/home/ahmad/Avatar/MachineLearning/data_prep_v1/shape_predictor_68_face_landmarks.dat'
        
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(dlibPredPath)
        if os.path.exists(os.path.join(predPath,outputDir)):
                print("output dir already exists. Deleting the existing folder...")
                shutil.rmtree(os.path.join(predPath,outputDir))
                
        os.mkdir(os.path.join(predPath,outputDir))
        
        fileList=[]
        imgList=[]
        images=[]
        for (dirpath, dirnames, filenames) in walk(predPath):
            
            fileList.extend(filenames)
            fileList=natsorted(fileList, alg=ns.IGNORECASE)
            for element in fileList:
                if ".jpg" in element:
                    imgList.append(element)
            for image in imgList:
                img = cv2.imread(os.path.join(predPath,image))
                if img is not None:
                    images.append(img)
        print(str(len(images)) + " frames recieved...")
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
    
                crop_img = cv2.resize(crop_img,(224,224))
                lipsROI_list.append(crop_img)
            except:
                print("ERROR")
        #Delete processes files
        for file in fileList:
            try:
                os.remove(os.path.join(predPath,file))
            except:
                pass
####### Concatnate the extracted faces and save in a seperate directory #######
        padding = 0
        max_width = []
        max_height = 0
        for img in lipsROI_list:
            max_width.append(img.shape[0])
            max_height += img.shape[1]
        w = np.max(max_width)
        h = max_height + padding
        # create a new array with a size large enough to contain all the images
        final_image = np.zeros((h, w, 3), dtype=np.uint8)

        current_y = 0  # keep track of where your current image was last placed in the y coordinate
        for image in lipsROI_list:
            # add an image to the final array and increment the y coordinate
            final_image[current_y:image.shape[0] + current_y, :image.shape[1], :] = image
            current_y += image.shape[0]
        #Use final_image for making predictions
        return final_image
    
    def callback(self, ros_data):
        
        
        if VERBOSE :
            print ('Received Prediction_Flag: "%s"' % ros_data)
        
        self.predictionFlag = ros_data.data       
        if (self.prev_predictionFlag==0) & (self.predictionFlag==0) :
            print("Receiving transmission from remote client...")
            self.prev_predictionFlag = self.predictionFlag
            
        if (self.prev_predictionFlag==0) & (self.predictionFlag==1) :
            print("Data recieved, analysing and performing prediction...")
            img = self.prepareData()
            predictionClass = LipReading_Predictor.performPrediction(img)
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
    '''Initializes and cleanup ros node'''
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
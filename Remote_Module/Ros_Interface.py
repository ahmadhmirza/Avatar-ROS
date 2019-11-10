#!/usr/bin/env python
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
------------------------------------------------------------------------------
Script to make a ROS Subscriber node to read frames from the android device.
------------------------------------------------------------------------------
Created on Fri Oct 11 11:11:11 2019
@author: Ahmad Hassan Mirza - ahmadhassan.mirza@gmail.com
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

__version__=  '0.1'
# Python libs
import sys, time

# numpy and scipy
import numpy as np
from scipy.ndimage import filters

# OpenCV
import cv2

# Ros libraries
import roslib
import rospy
import os

# Ros Messages
from sensor_msgs.msg import CompressedImage
# We do not use cv_bridge it does not support CompressedImage in python
# from cv_bridge import CvBridge, CvBridgeError
VERBOSE=False
    
    

class image_feature:
        

    def __init__(self):
        '''Initialize ros subscriber'''
        global imageList
        global count 
        global image_path
        image_path = r'/home/ahmad/Avatar/Prediction_Path'
        count = 0
        imageList=[]
        # subscribed Topic
        self.subscriber = rospy.Subscriber("/image_transport",
            CompressedImage, self.callback,  queue_size = 1)
        
        if VERBOSE :
            print ("subscribed to /image_transport")


    def callback(self, ros_data):
        global imageList
        global count
        '''Callback function of subscribed topic. 
        Here images get converted and features detected'''
        #print("here")
        if VERBOSE :
            print ('received image of type: "%s"' % ros_data.format)

        #### direct conversion to CV2 ####
        np_arr = np.frombuffer(ros_data.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        #image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR) # OpenCV >= 3.0:
        
        #### Feature detectors using CV2 #### 
        # "","Grid","Pyramid" + 
        # "FAST","GFTT","HARRIS","MSER","ORB","SIFT","STAR","SURF"
       # method = "GridFAST"
        
        #feat_det = cv2.xfeatures2d.SIFT_create()
        
        print("Frame Received..")
        # convert np image to grayscale
        #featPoints = feat_det.detect(
            #cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY))
        
        #time2 = time.time()
        #if VERBOSE :
            #print ('%s detector found: %s points in: %s sec.'%(method,
               # len(featPoints),time2-time1))
        
        #for featpoint in featPoints:
            #x,y = featpoint.pt
            #cv2.circle(image_np,(int(x),int(y)), 3, (0,0,255), -1)
        imageList.append(image_np)
        #cv2.imshow('cv_img', image_np)
        #image_path = r'/home/ahmad/Avatar/MachineLearning/data_prep_v1/My_Training_Data/Daniel/p03/01'
        
        if len(imageList) == 14:
            print("Images in list: " + str(len(imageList)))
            for img in imageList:
                count+=1
                #time1 = time.time()
                imageName = "frame%d.jpg" % count
                Img_path = os.path.join(image_path,imageName)
                cv2.imwrite(Img_path, img)
                print("Frames Saved..")
            imageList.clear()
        #cv2.waitKey(1)

# # =============================================================================
#         #### Create CompressedIamge ####
#         msg = CompressedImage()
#         msg.header.stamp = rospy.Time.now()
#         msg.format = "jpeg"
#         msg.data = np.array(cv2.imencode('.jpg', image_np)[1]).tostring()
#         # Publish new image
#         self.image_pub.publish(msg)
#         
#         #self.subscriber.unregister()
# =============================================================================

def main(args):
    '''Initializes and cleanup ros node'''
    ic = image_feature()
    rospy.init_node('Avatar_Desktop', anonymous=False)
    print("ok")
    try:
        rospy.spin()
        
    except KeyboardInterrupt:
        print ("Shutting down ROS Image feature detector module")
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(sys.argv)

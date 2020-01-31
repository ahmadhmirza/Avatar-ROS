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

    '''Callback function of subscribed topic. 
    Here images are received and converted and saved to disk'''
    def callback(self, ros_data):
        global imageList
        global count
        BATCH_SIZE = 14
        if VERBOSE :
            print ('received image of type: "%s"' % ros_data.format)

        #### conversion to cv2 compatible format ####
        np_arr = np.frombuffer(ros_data.data, np.uint8)
        image_np = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)    
        print("Frame Received..")
        
        imageList.append(image_np)
        # The images are written to disk after receiving the images
        # defined by BATCH_SIZE
        if len(imageList) == BATCH_SIZE:
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

"""
Main class creates an object of image_feature class and hands control over 
to rospy.spin function which keeps calling running the methods in this class
in a loop
"""
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

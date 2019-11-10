#!/usr/bin/env python
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
------------------------------------------------------------------------------
Script to make prediction once the transmission from Remote device is complete
------------------------------------------------------------------------------
Created on Sun Nov 10 10:10:10 2019
@author: Ahmad Hassan Mirza - ahmadhassan.mirza@gmail.com
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

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
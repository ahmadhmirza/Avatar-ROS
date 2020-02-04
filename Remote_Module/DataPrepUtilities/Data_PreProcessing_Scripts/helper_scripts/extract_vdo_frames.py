#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  2 09:06:07 2019

@author: ahmad
"""

import cv2
import os

import numpy as np

def rotateImage(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result


pathPrefix = r'/home/ahmad/Avatar/MachineLearning/data_prep_v1/My_Training_Data/'
pathPrefix_1 = r'Martin/'
pathPrefix_2 = r'phrases/p03/'
fileName = "Thank_you.mp4"

path = os.path.join(pathPrefix,pathPrefix_1,pathPrefix_2,fileName)

vidcap = cv2.VideoCapture(path)
success,image = vidcap.read()
count = 0
while success:
  #image = rotateImage(image,90) 
  cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file      
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1
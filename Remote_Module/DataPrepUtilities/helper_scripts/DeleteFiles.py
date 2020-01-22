"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
------------------------------------------------------------------------------
This script will delete all the files in the path.
Intended to be run after the face/lips roi images have been extracted and saved
in a seperate location.
Afterwards this script can be run to delete the original unwanted images
------------------------------------------------------------------------------
Created on Fri Oct 11 11:11:11 2019
@author: Ahmad Hassan Mirza - ahmadhassan.mirza@gmail.com
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#import cv2
#import dlib
import os
from os import walk

def initilizeDirList(dirListFile):
    # Open the file with read only permit
    f = open(dirListFile, "r")
    # use readlines to read all lines in the file
    # The variable "lines" is a list containing all lines in the file
    dirList = f.readlines()
    # close the file after reading the lines.
    f.close()
    return dirList
    
def loadImagesFromDir(path):
    fileList=[]
    #imgList=[]
    #images=[]
    for (dirpath, dirnames, filenames) in walk(path):
        #genDirectoryList.extend(dirnames)
        fileList.extend(filenames)
    return fileList
    
global dirList
inputFile = r'C:\Users\MIR6SI\Desktop\data_prep\config_files\image_paths.txt'

dirList = initilizeDirList(inputFile)
####### Code for OpenCV #####
pathIndex=0
for path in dirList:    
    pathIndex += 1
    path = path.replace("\\", "\\\\").strip()
    if os.path.exists(path):
        #print(path)
        totalDirs = len(dirList)
        fileList = loadImagesFromDir(path)
        percentComplete = (pathIndex/totalDirs) * 100
        count = 0
        for file in fileList:
            count +=1
            try:
                os.remove(os.path.join(path,file))
            except:
                pass
    print("Progress = " + str(round(percentComplete,3))+ "%")
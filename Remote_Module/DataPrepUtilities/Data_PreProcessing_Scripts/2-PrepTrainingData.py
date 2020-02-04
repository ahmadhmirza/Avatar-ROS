"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
------------------------------------------------------------------------------
Script to move the concatnated faceROI image to their respective folders in the
Arranged Training Data directory so they can be mapped to their classes.
Input text file containng destination and source paths is required.
text file should be in csv format
------------------------------------------------------------------------------
Created on Fri Oct 11 11:11:11 2019
@author: Ahmad Hassan Mirza - ahmadhassan.mirza@gmail.com
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import os
import csv
from os import walk
from shutil import copyfile
import cv2

def initilizeDirList(dirListFile):
    sourceDestMapping=[]
    with open(dirListFile) as csvPathsFile:
        csvReader = csv.reader(csvPathsFile)
        for row in csvReader:
            path=[]
            if "#" in row[0]:
                pass
            else:
                path.append(row[0])
                path.append(row[1])
                sourceDestMapping.append(path)
    return sourceDestMapping

def loadImagesFromDir(path):
    fileList=[]
    imgList=[]
    images=[]
    for (dirpath, dirnames, filenames) in walk(path):
        #genDirectoryList.extend(dirnames)
        fileList.extend(filenames)
        for element in fileList:
            if ".jpg" in element:
                imgList.append(element)
        for image in imgList:
            img = cv2.imread(os.path.join(path,image))
            if img is not None:
                images.append(img)
        return images
    
def readCsv(dirListFile):
    sourceDestMapping=[]
    with open(dirListFile) as csvPathsFile:
        csvReader = csv.reader(csvPathsFile)
        for row in csvReader:
            path=[]
            if "#" in row[0]:
                pass
            else:
                path.append(row[0])
                path.append(row[1])
                sourceDestMapping.append(path)
    return sourceDestMapping

################################### Main ######################################
    
global dirList

###### Paths for Windows #######
inputFile = r'/home/ahmad/Avatar/MachineLearning/data_prep_v1/config_files/Arrange_paths.csv'
configFilesDirectory = r'/home/ahmad/Avatar/MachineLearning/data_prep_v1/config_files/'

# ###### Paths for Linux #######
# =============================================================================
# inputFile = r'C:\Users\MIR6SI\Desktop\data_prep\config_files\Arrange_paths.csv'
# mappingsFile = r'C:\Users\MIR6SI\Desktop\data_prep\config_files\mapping.csv'
# configFilesDirectory = r'C:\Users\MIR6SI\Desktop\data_prep\config_files'
# =============================================================================

sdMap = initilizeDirList(inputFile)
sourcePrefix = [["01/TrainingData/Concatnated_Images","\\"],
                ["02/TrainingData/Concatnated_Images","\\"],
                ["03/TrainingData/Concatnated_Images","\\"],
                ["04/TrainingData/Concatnated_Images","\\"],
                ["05/TrainingData/Concatnated_Images","\\"],
                ["06/TrainingData/Concatnated_Images","\\"],
                ["07/TrainingData/Concatnated_Images","\\"],
                ["08/TrainingData/Concatnated_Images","\\"],
                ["09/TrainingData/Concatnated_Images","\\"],
                ["10/TrainingData/Concatnated_Images","\\"]]

############################## Block -1 #######################################
# Copy data to training data folder #
currentDir=0
totalDirectories=len(sdMap)
count = 0
for dir in sdMap:
    currentDir += 1
    #destPath = dir[0].replace("\\", "\\\\").strip()
    destPath = dir[0]
    #print(destPath)
    #sourcePath = dir[1].replace("\\", "\\\\").strip()
    sourcePath = dir[1]
    if os.path.exists(destPath):
        if os.path.exists(sourcePath):
            
            for subDir in sourcePrefix:
                sourcePathFull = os.path.join(sourcePath,subDir[0])
                
                fileList=[]
                
                for (dirpath, dirnames, filenames) in walk(sourcePathFull):
                    #genDirectoryList.extend(dirnames)
                    fileList.extend(filenames)
                    
                for file in fileList:
                    count +=1
                    print(count)
                    sourceFilePath= os.path.join(sourcePathFull,file)
                    destFilePath= os.path.join(destPath,file)
                    copyfile(sourceFilePath,destFilePath)
                    
                    new_file_name = "ctcImage_"+str(count)+".jpg"
                    new_dst_file_name = os.path.join(destPath, new_file_name)
                    os.rename(destFilePath, new_dst_file_name)
                    
    progress = currentDir*100/totalDirectories
    print("Progress(BLOCK-1): " + str(round(progress,3)))

print(" Training Data folder is ready...")
print("========================END OF Script========================")
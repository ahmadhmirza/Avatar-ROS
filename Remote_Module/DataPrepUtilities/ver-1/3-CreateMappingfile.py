

import os
import csv
from os import walk
from shutil import copyfile
import cv2
from natsort import natsorted, ns

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
        fileList=natsorted(fileList, alg=ns.IGNORECASE)
        for element in fileList:
            if ".jpg" in element:
                imgList.append(element)
        for image in imgList:
            img = cv2.imread(os.path.join(path,image))
            if img is not None:
                images.append(img)
        return images
def getFileNames(path):
    fileList=[]
    imgList=[]
    for (dirpath, dirnames, filenames) in walk(path):
        #genDirectoryList.extend(dirnames)
        fileList.extend(filenames)
        fileList=natsorted(fileList, alg=ns.IGNORECASE)
        for element in fileList:
            if ".jpg" in element:
                imgList.append(element)
    return imgList

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
############################## Block -2 #######################################
# read images into a list
# map the images to their class   
config_files = r"/home/ahmad/Avatar/MachineLearning/data_prep_v1/config_files"
classMaps = r"/home/ahmad/Avatar/MachineLearning/data_prep_v1/config_files/mapping.csv"
basePath = r'/home/ahmad/Avatar/MachineLearning/data_prep_v1/Arranged_training_data'   
            


MapFileName="ImageToClassMappings.csv"
MapFilePath = os.path.join(config_files,MapFileName)

classMapings = readCsv(classMaps)

try:
     MappingsFile= open(MapFilePath,"w+")
except:
     print("unable to create Mapping.csv file")
 
MappingsFile.write("Image_ID,Class\n")

classList=["w01","w02","w03","p01","p02","p03"]
for item in classList:
    imgPath = os.path.join(basePath,item)
    print("Checking in Path: " + imgPath)
    imgList = getFileNames(imgPath)
    print(imgList)
    for img in imgList:
        for x in classMapings:
            if "#" in x[0]:
                pass
            elif item in x:
                MappingsFile.write(img + "," + x[1]+"\n")

print("Script Executed...")
MappingsFile.close()
###############################################################################
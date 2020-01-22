"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
------------------------------------------------------------------------------
This script will extract the face ROI from all the images available in the path
provided in the text file.
The extracted face ROI will be saved as a new image in the directory called
'TrainingData' with in the path where the original images are
Steps Performed:
    1- Extract Faces.
    2- Detele original images.
    3- Concatnate faceROI images into one big image.

Script revamped for Linux
------------------------------------------------------------------------------
Created on Fri Oct 11 11:11:11 2019
@author: Ahmad Hassan Mirza - ahmadhassan.mirza@gmail.com
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
import cv2
import os
from os import walk
import shutil
import matplotlib.pyplot as plt
from natsort import natsorted, ns
import numpy as np
##################
import dlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
################################
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

def getFileList(path):
    fileList=[]
    #imgList=[]
    #images=[]
    for (dirpath, dirnames, filenames) in walk(path):
        #genDirectoryList.extend(dirnames)
        fileList.extend(filenames)
    return fileList
##############################################################################
#Displays the image in the iPython Console
##############################################################################
def displayImage(img):
    plt.imshow(img)
    plt.show()

def concatnateImages(images):
        padding = 0
        max_width = []
        max_height = 0
        for img in images:
            max_width.append(img.shape[0])
            max_height += img.shape[1]
        w = np.max(max_width)
        h = max_height + padding
        # create a new array with a size large enough to contain all the images
        final_image = np.zeros((h, w, 3), dtype=np.uint8)

        current_y = 0  # keep track of where your current image was last placed in the y coordinate
        for image in images:
            # add an image to the final array and increment the y coordinate
            final_image[current_y:image.shape[0] + current_y, :image.shape[1], :] = image
            current_y += image.shape[0]
        return final_image
        
######## Functions to extract face from images and save them in a dir #########
def convertToRGB(img): 
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def detect_faces(f_cascade, colored_img, scaleFactor = 1.1):
    #just making a copy of image passed, so that passed image is not changed
    gray = colored_img.copy()
    #convert the test image to gray image as opencv face detector expects gray images
    #gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY) 
    #displayImage(gray)      
    #let's detect multiscale (some images may be closer to camera than others) images
    faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5) 
    return faces

# Function extracts Lips' ROI from the image and return the cropped image
def detect_lips(detector,predictor,img,currentPath):
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
        
        #print("x1 :" +str(x1) + ", y1 :" +str(y1))
        #print("x2 :" +str(x2) + ", y2 :" +str(y2))
        
        #print("width :" +str(w) + ", height :" +str(h))
        
        h=y2-y1
        w=x2-x1
        
        
        #print("width :" +str(w) + ", height :" +str(h))
        #print("x+w :" +str(x1+w) + ", y+h :" +str(y1+h))
        crop_img = img[y1:y1+h, x1:x1+w]
    
        #plt.imshow(crop_img)
        #plt.show()
        crop_img = cv2.resize(crop_img,(224,224))
        return crop_img
    except:
        print("ERROR: " + currentPath)
        displayImage(img)
        return img

#applies dct on the provided image
# scale = 255.0
def applyDCT(img,scale):
    #img = cv2.imread(fn, 0)      # 1 chan, grayscale!
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imf = np.float32(img)/scale  # float conversion/scale
    dst = cv2.dct(imf)           # the dct
    img = np.uint8(dst)*scale    # convert back
    return img

def applyCannyEdge(img):
    displayImage(img)
    edges = cv2.Canny(img,100,200)
    displayImage(edges)
    #return edges
    
def cropImage(img,x,y,w,h):
    return img[y:y+h, x:x+w]

################################# MAIN #######################################
#=============Input Arguments Handling================

global dirList
#inputFile = r'/home/ahmad/Avatar/MachineLearning/data_prep_v1/config_files/image_paths_v2.txt'

#inputFile = r'/home/ahmad/Avatar/MachineLearning/data_prep_v1/config_files/image_paths.txt'

inputFile = r'/home/ahmad/Avatar/MachineLearning/data_prep_v1/config_files/image_paths_test_data.txt'
outputDir = 'TrainingData'
concatDir = "Concatnated_Images"  
dirList = initilizeDirList(inputFile)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

####### Code for OpenCV #####
pathIndex=0
for path in dirList:    
    pathIndex += 1
    path = path.strip("\n")
    if os.path.exists(os.path.join(path,outputDir)):
        print("output dir already exists. Deleting the existing folder...")
        shutil.rmtree(os.path.join(path,outputDir))
    #path = path.replace("\\", "\\\\").strip()
    os.mkdir(os.path.join(path,outputDir))
    #print("Output Dir created successfully...")
# =============================================================================
#     except:
#         print("Unable to create output directory")  
# =============================================================================
    totalDirs = len(dirList)
    percentComplete = (pathIndex/totalDirs) * 100
    images=loadImagesFromDir(path)
    #load cascade classifier training file for haarcascade
    try:
        haar_face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    except:
        print("haarcascade_frontalface_alt.xml not found in script path")
        #exit(1)
        
    imageIndex = 0
    for img in images:
        #plt.imshow(convertToRGB(img))
        imageIndex += 1
        #call our function to detect faces
        #print("Detecting Face in image: " + str(imageIndex))
        faces_detected = detect_faces(haar_face_cascade, img)
        #go over list of faces and draw them as rectangles on original colored img
        for (x, y, w, h) in faces_detected:
            #cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            #print("Extracting face ROI")
            test_cropped = cropImage(img,x,y,w,h)     
        imageName= os.path.join(path, outputDir + "/TestData_" + str(imageIndex)+ ".jpg")
        #print("Saving Image in the output dirctory")
        test_cropped = detect_lips(detector,predictor,test_cropped,imageName)
        #test_cropped = applyDCT(test_cropped,50.0)
        #displayImage(test_cropped)
        cv2.imwrite(imageName,test_cropped)
        fileList=getFileList(path)
        
        #######################################################################
        # Block to extract lips from the FaceROI #
        # This will be the final training data #
        #######################################################################
        
    #Delete processed files to free up disk space
    #for file in fileList:
    #    try:
     #       os.remove(os.path.join(path,file))
     #   except:
     #       pass
    
####### Concatnate the extracted faces and save in a seperate directory #######
    # Path for extracted faces    
    trainingDataPath = os.path.join(path, outputDir) 
    # Path where the concatnated image is to be saved
    concatDataPath = os.path.join(trainingDataPath,concatDir)
    if os.path.exists(concatDataPath):
        print("output dir already exists. Deleting the existing folder...")
        shutil.rmtree(concatDataPath)
    os.mkdir(concatDataPath)
    #load the extracted faces images
    imgList=loadImagesFromDir(trainingDataPath)
    #print("Extracted Faces Images: " +str(len(imgList)))
    conImage = concatnateImages(imgList)
    #displayImage(conImage)
    imageName = os.path.join(concatDataPath,"cctImage.jpg")
    cv2.imwrite(imageName,conImage)
    print("Progress = " + str(round(percentComplete,3)))
    
    
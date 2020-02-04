import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np    # for mathematical operations
#import tensorflow.compat.v1 as tf
import tensorflow as tf
tf.disable_v2_behavior()


def showProgress(current,total,blockName):
    progress = current*100/total
    print("Progress("+str(blockName)+"): " + str(round(progress,2)))   
    
basePath = r'/home/ahmad/Avatar/MachineLearning/data_prep_v1/Arranged_training_data/1_Final_Data/'   
csvFile = r'/home/ahmad/Avatar/MachineLearning/data_prep_v1/config_files/ImageToClassMappings.csv'
data = pd.read_csv(csvFile)

# =============================================================================
# The mapping file contains two columns:
# Image_ID: Contains the name of each image
# Class: Contains corresponding class for each image

#Following block reads the images based on their names i.e Image_ID column. 
# =============================================================================
X = [ ]     # creating an empty array
total = len(data.Image_ID)
print(total)
current = 0
for img_name in data.Image_ID:
    current += 1
    try:
        img = plt.imread(os.path.join(basePath + img_name))
        #print(os.path.join(basePath + img_name))
        X.append(img)  # storing each image in array X
        showProgress(current,total,"Block-1")
        
    except:
        print("exception")
        showProgress(current,total,"Block-1")    
X = np.array(X)    # converting list to array

print (len(X))

# =============================================================================
# we need two things to train our model:
# Training images, and
# Their corresponding class
# =============================================================================
from keras.utils import np_utils
y = data.Class
dummy_y = np_utils.to_categorical(y)    # one hot encoding Classes

print("Encoding Class : (dummy_y) :  Ready")

# =============================================================================
# We will be using a VGG16 pretrained model which takes an input image of shape 
# (224 X 224 X 3). Since our images are in a different size, we need to reshape 
# all of them. We will use the resize() function of skimage.transform
# =============================================================================
from skimage.transform import resize 

img = []
print ("Resizing Images...")
total = len(X)
current = 0
for i in range(0,X.shape[0]):
    current +=1
    a = resize(X[i], preserve_range=True, output_shape=(224,224)).astype(int)      # reshaping to 224*224*3
    img.append(a)
    showProgress(current,total,"Resizing Op")
X = np.array(img)
print ("Images Resized...")

##############################################################################
# =============================================================================
#  Before passing any input to the model, we must preprocess it as per the
#  model’s requirement. Otherwise, the model will not perform well enough
# =============================================================================

print ("Pre-processing Images...")
from keras.applications.vgg16 import preprocess_input
X = preprocess_input(X, mode='tf')      # preprocessing the input data

print("Pre-Processing Done...")

print("Preparing train and test data..") 
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, dummy_y, test_size=0.3, random_state=42)    # preparing the validation set

print("Data prepared...")

##############################################################################

#Build the Model
#Using the VGG16 pretrained model
import keras
from keras.models import Sequential
from keras.applications.vgg16 import VGG16
#from keras.layers import Dense, InputLayer, Dropout
from keras.layers import Dense, InputLayer
print(keras.__version__)
modelPath = r"/home/ahmad/Avatar/MachineLearning/data_prep_v1/1_Trained_Model"
modelName = "Avatar_lr_Model.h5"
TF_Model = os.path.join(modelPath,modelName)
#from keras.applications import resnet50
from pypac import pac_context_for_url
with pac_context_for_url("https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

#Loading VGG16 model and saving it as base_model

print("VGG Model loaded...")   

############################################################################### 
# include_top=False to remove the top layer

# Make predictions using this model for X_train and X_valid, get the features, 
#and then use those features to retrain the model.
###############################################################################


print("Making predictions...")
X_train = base_model.predict(X_train,batch_size = 20)
X_valid = base_model.predict(X_valid,batch_size = 20)
X_train.shape, X_valid.shape

print(X_train.shape)
print(X_valid.shape)

# Shape of X_train and X_valid is 574, 7, 7, 512), (246, 7, 7, 512) respectively.
# In order to pass it to our neural network, we have to reshape it to 1-D.
print("Converting base models to 1-D")
###change expected here###
X_train = X_train.reshape(619, 7*7*512)      # converting to 1-D
X_valid = X_valid.reshape(266, 7*7*512)
print("1-D Conversion : Done")

###############################################################################
print("preprocessing images to zero center...")
#preprocess the images and zero-center them -> helps the model converge faster
train = X_train/X_train.max()      # centering the data
X_valid = X_valid/X_train.max()
print("Zero Centering: Done")
##############################################################################
#Build the model in 3steps:
# 1.Building the model
# 2.Compiling the model
# 3.Training the model

print("Building the model-Stage1")
# =============================Stage -1 =======================================
model = Sequential()
model.add(InputLayer((7*7*512,)))    # input layer
model.add(Dense(units=1024, activation=tf.nn.relu,input_shape=(7*7*512,))) # hidden layer
model.add(Dense(7,input_shape=(1024,), activation='softmax'))    # output layer

model.summary()
print("Stage-1: Done")
###############################################################################
# =============================Stage -2 =======================================
print("===Stage2===")
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print("Stage-2: Done")
#############################################################################
# =============================Stage -3 =======================================
print("===Stage3===")
#epoch 87/97 - gave max val_accuracy
model.fit(train, y_train, epochs=100, validation_data=(X_valid, y_valid))
print("Stage-3: Done")
print("Building the Model : Done")

model.summary()

#evaluate the model

scores = model.evaluate(train,y_train,verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

###############################################################################
# =============================Stage -3.1 =======================================
print("Saving the Model")

from keras.models import model_from_json

modelJson = r'/home/ahmad/Avatar/MachineLearning/data_prep_v1/TrainedModel/model.json'
modelH5 = r'/home/ahmad/Avatar/MachineLearning/data_prep_v1/TrainedModel/model.h5'

# serialize model to JSON
model_json = model.to_json()
with open(modelJson, "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(modelH5)
print("Saved model to disk")
#model.save(TF_Model)
#
#
## Recreate the exact same model, including its weights and the optimizer
#new_model = tf.keras.models.load_model(TF_Model)
## Show the model architecture
#new_model.summary()
#loss, acc = new_model.evaluate(X_valid,  y_valid, verbose=2)
#print('Restored model, accuracy: {:5.2f}%'.format(100*acc))

#############################################################################
print("====================Ready to test Lip Reading Model====================")
#import cv2

NumToLang = {
        0:"Un-Classified",
        1:"Start",
        2:"Stop",
        3:"Hello",
        4:"How are you?",
        5:"Nice to Meet you",
        6:"Thank you"
        }

print("***** External Videos Test *****")
print("***** Ready for testing *****")


TestImage_list=[
    r'/home/ahmad/Avatar/MachineLearning/data_prep_v1/Arranged_training_data/2_Test_Data/p01/1.jpg',
    r'/home/ahmad/Avatar/MachineLearning/data_prep_v1/Arranged_training_data/2_Test_Data/p01/2.jpg',
    r'/home/ahmad/Avatar/MachineLearning/data_prep_v1/Arranged_training_data/2_Test_Data/p01/3.jpg',
    r'/home/ahmad/Avatar/MachineLearning/data_prep_v1/Arranged_training_data/2_Test_Data/p01/4.jpg',
    r'/home/ahmad/Avatar/MachineLearning/data_prep_v1/Arranged_training_data/2_Test_Data/p01/5.jpg',
    r'/home/ahmad/Avatar/MachineLearning/data_prep_v1/Arranged_training_data/2_Test_Data/p02/1.jpg',
    r'/home/ahmad/Avatar/MachineLearning/data_prep_v1/Arranged_training_data/2_Test_Data/p02/2.jpg',
    r'/home/ahmad/Avatar/MachineLearning/data_prep_v1/Arranged_training_data/2_Test_Data/p02/3.jpg',
    r'/home/ahmad/Avatar/MachineLearning/data_prep_v1/Arranged_training_data/2_Test_Data/p02/4.jpg',
    r'/home/ahmad/Avatar/MachineLearning/data_prep_v1/Arranged_training_data/2_Test_Data/p02/5.jpg',
    r'/home/ahmad/Avatar/MachineLearning/data_prep_v1/Arranged_training_data/2_Test_Data/p03/1.jpg',
    r'/home/ahmad/Avatar/MachineLearning/data_prep_v1/Arranged_training_data/2_Test_Data/p03/2.jpg',
    r'/home/ahmad/Avatar/MachineLearning/data_prep_v1/Arranged_training_data/2_Test_Data/p03/3.jpg',
    r'/home/ahmad/Avatar/MachineLearning/data_prep_v1/Arranged_training_data/2_Test_Data/p03/4.jpg',
    r'/home/ahmad/Avatar/MachineLearning/data_prep_v1/Arranged_training_data/2_Test_Data/p03/5.jpg']

        
count = 0
for images in TestImage_list:
    test_image = []
    img = plt.imread(images)
    test_image.append(img)
    test_img = np.array(test_image)
    count+=1
    
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
    prediction = model.predict_classes(test_image,batch_size=10)
    print(prediction)
    
    print("Prediction: Person " + str(count) + " said : " + NumToLang.get(prediction[0]))
    
print("============================Testing with saved model============================")  

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = model.evaluate(train,y_train,verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

   
for images in TestImage_list:
    test_image = []
    img = plt.imread(images)
    test_image.append(img)
    test_img = np.array(test_image)
    count+=1
    
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
    prediction = loaded_model.predict_classes(test_image,batch_size=10)
    print(prediction)
    
    print("Prediction: Person " + str(count) + " said : " + NumToLang.get(prediction[0]))
  
print("============================End of Script============================")
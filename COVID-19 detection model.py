#Importing libraries

import numpy as np
import keras
import csv
import glob
import pandas as pd
import seaborn as sn
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Activation, Dropout, MaxPooling2D, BatchNormalization, ZeroPadding2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras import optimizers
from PIL import Image as img
from matplotlib.pyplot import imshow
from IPython.display import Image 
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.models import Model
from numpy import expand_dims
%matplotlib inline

#Train, test and validation sets loading.
#First, the train set will be loaded in the x_train variables, which will contain the images in format
# np.array, and y_train, which will contain the categories to which the images belong.

# Therefore, we open the text file that contains the path of the images and their labels:

with open('training.txt') as train_fich:
    train_reader = csv.reader(train_fich, delimiter="\t")
    train_fich_list = list(train_reader)
    
#We save the content of the file in a list of lists, in train_fich_list [n] [0] the labels will be saved
#of the images and in train_fich_list [n] [1] the path of the images will be saved, where n is the number of elements
#contents in 'training.txt'

#We save the content of the file in a list of lists, in train_fich_list [n] [0] the labels will be saved
#of the images and in train_fich_list [n] [1] the path of the images will be saved, where n is the number of elements
#contents in 'training.txt'

#With the following line, we separate the labels from the images and save them in a list.
y_train=[i[0] for i in train_fich_list]
#And we get the paths of the images that make up the train set.
x_train_path=[i[1]for i in train_fich_list]

#Now we convert the stored paths into images, for this we resort to the "open" function of 
# PIL module, which adds image processing capabilities to the Python interpreter. It creates a loop that goes
#through the list where the paths are stored, the paths are converted into images, the size is adjusted to the
#dimensions that we need and these images are in turn in np.array elements that will finally be stored in one
#list.

#The size of the images is fixed at 224x224 since AlexNet (will be analyzed in the next section)
#use images from this dimesion as input to CNN.
i=0
x_train_list=[]
l=len(x_train_path)
for i in range(l):
    img1 =img.open(x_train_path[i])
    img1=img1.resize((224,224))
    img1=img1.convert(mode='RGB')
    x_train_list.append(np.array(img1))
    i+=1

#The result of the previous function is that we will have a list of np.array elements. Last step will be convert this list to np.array 
#format and normalize its elements. Additionally, it will be normalized and OneHotEncoder will be applied.
x_train=np.array(x_train_list)/255
y_train=np.array(y_train)
y_train=to_categorical(y_train,3)

#We show the first image in the set
plt.imshow(x_train[0],cmap='gray')

#Repeat the process for the test set:
#Therefore, we open the text file that contains the paths or paths of the images and their labels:
with open('testing.txt') as test_fich:
    test_reader = csv.reader(test_fich, delimiter="\t")
    test_fich_list = list(test_reader)
#We save the content of the file in a list of lists, in test_fich_list [n] [0] the labels will be saved
#of the images and in test_fich_list [n] [1] the path of the images will be saved, where n is the number of elements
#contents in 'testing.txt'

#With the following line, we separate the labels from the images and save them in a list type variable.
y_test = [j [0] for j in test_fich_list]
#And we get the paths of the images that make up the test set.
x_test_path = [j [1] for j in test_fich_list]
#Check that both vectors are of the correct length:
len (y_test), len (x_test_path)

#Now we must convert the stored paths into images, just like we did for the train set
j = 0
x_test_list = []
l = len (x_test_path)
for j in range (l):
     img1 = img.open (x_test_path [j])
     img1 = img1.resize ((224,224))
     img1 = img1.convert (mode = 'RGB')
     x_test_list.append (np.array (img1))
     j + = 1

#The result of the previous function is that we will have a list of np.array elements. Last step will be convert this list to np.array 
#format and normalize its elements. Additionally, it is also will normalize and apply a OneHotEncoder on the variable y_test.
x_test = np.array (x_test_list) / 255
y_test = np.array (y_test)
y_test = to_categorical (y_test, 3)

#We show the first image in the test set
plt.imshow(x_test[0],cmap='gray')

#And finally, we form the validation set:

#Therefore, we open the text file that contains the paths or paths of the images and their labels:
with open ('validation.txt') as val_fich:
     val_reader = csv.reader (val_fich, delimiter = "\ t")
     val_fich_list = list (val_reader)
#We save the content of the file in a list of lists, in val_fich_list [n] [0] the labels will be saved
#of the images and in the val_fich_list [n] [1] the path of the images will be saved, where n is the number of elements
#contents in 'validation.txt'

#With the following line, we separate the labels from the images and save them in a list type variable.
y_val = [z [0] for z in val_fich_list]
#And we get the paths of the images that make up the validation set.
x_val_path = [z [1] for z in val_fich_list]

#Now we must convert the stored paths into images, just like we did for the validation set
z = 0
x_val_list = []
l = len (x_val_path)
for z in range (l):
     img1 = img.open (x_val_path [z])
     img1 = img1.resize ((224,224))
     img1 = img1.convert (mode = 'RGB')
     x_val_list.append (np.array (img1))
     z + = 1
 
#The result of the previous function is that we will have a list of np.array elements. Last step will be convert this list to 
#np.array convert this list to np.array format and normalize its elements. Additionally, it is also will normalize and apply a OneHotEncoder on the variable y_val.
x_val = np.array (x_val_list) / 255
y_val = np.array (y_val)
y_val = to_categorical (y_val, 3)

#We show the first image of the validation set:
plt.imshow (x_val [0], cmap = 'gray')


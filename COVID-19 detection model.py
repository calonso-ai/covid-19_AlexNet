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

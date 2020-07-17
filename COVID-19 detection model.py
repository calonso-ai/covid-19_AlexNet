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

#Model creation and training pipeline

# According to the study indicated in reference [3], where the capacities analysis of 10 CNN is carried out
# on a set of medical images of patients with COVID-19 and healthy patients, the best neural network
#results presented is AlexNet. In this section, two CNNs will be designed, a simple CNN network and an AlexNet network
#in order to evaluate the results of both methods. Additionally, the influence of applying
# Online Data Augmentation techniques on the train set is analysed.

#Simple CNN creation

#A simple CCN is created, that is, with few layers in both the convolutional stage and the stage
#FC. In the convolutional stage a convolutional layer is defined with 2 kernels of size 2x2.
# In the FC stage, a Flattan layer is defined that transforms a two-dimensional matrix (in this case, our
#imagenes) in a vector that can be processed by the output Dense layer, in this case with 3 nodes, since the
# classification will be made between 3 categories.
def create_cnn ():
    model=Sequential()
    model.add(Conv2D(2, (2, 2), activation='relu', input_shape=(224,224, 3)))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))
    return model

# A simple CNN is created, according to the model defined above, using Adam as an optimizer.
model1=create_cnn()
model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Next, the model is trained using the validation set as the validation set.
hist_model1=model1.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=20)

#We are going to apply real time Data Augmentation, performing rotation, displacement and flip of images of the set
#de train.
train_aug = ImageDataGenerator (
     rotation_range = 20,
     width_shift_range = 0.2,
     height_shift_range = 0.2,
     horizontal_flip = True
)

#Simple CNN with online DATA AUGMENTATION

#To do this, a new model is created, symmetrical to the one used previously, since the objective is to analyze Data Augmentation influence
model2 = create_cnn ()
model2.compile (loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

#Next, we train the model using as validation set our validaton set and online application
#Data Augmentation to the train set.
hist_model2 = model2.fit (train_aug.flow (x_train, y_train), validation_data = (x_val, y_val), epochs = 20)

def create_AlexNet ():
    model = Sequential ()

    # First Convolutionary Layer
    
    #With 96 11x11 kernels, stride 4, enabling padding and with one input
    #de 224x224x3.
    model.add (Conv2D (filters = 96, input_shape = (224,224.3), kernel_size = (11.11), strides = (4.4), padding = 'valid'))
    model.add (Activation ('relu'))
    # Pooling layer, with a 2x2 kernel, 2 stride and padding enabled.
    model.add (MaxPooling2D (pool_size = (2,2), strides = (2,2), padding = 'valid'))
    # Batch Normalisation before moving on to the fifth layer.
    model.add (BatchNormalization ())

    # Second Convolutionary Layer
    
    #With 256 11x11 kernels, 4 stride, enabling padding.
    model.add (Conv2D (filters = 256, kernel_size = (11,11), strides = (1,1), padding = 'valid'))
    model.add (Activation ('relu'))
    # Pooling layer, with a 2x2 kernel, 2 stride and padding enabled.
    model.add (MaxPooling2D (pool_size = (2,2), strides = (2,2), padding = 'valid'))
    # Batch Normalisation before moving on to the fifth layer.
    model.add (BatchNormalization ())

    # Third Convolutionary Layer
    
    #With 384 3x3 kernels, stride of 1, enabling padding.
    model.add (Conv2D (filters = 384, kernel_size = (3,3), strides = (1,1), padding = 'valid'))
    model.add (Activation ('relu'))
    # Batch Standardization
    model.add (BatchNormalization ())

    # Fourth Convolutionary Layer
    
    #With 384 3x3 kernels, stride of 1, enabling padding.
    model.add (Conv2D (filters = 384, kernel_size = (3,3), strides = (1,1), padding = 'valid'))
    model.add (Activation ('relu'))
    # Batch Standardization
    model.add (BatchNormalization ())

    # Fifth Convolutionary Layer
    
    #With 256 3x3 kernels, stride of 1, enabling padding.
    model.add (Conv2D (filters = 256, kernel_size = (3,3), strides = (1,1), padding = 'valid'))
    model.add (Activation ('relu'))
    # Pooling layer, with a 2x2 kernel, 2 stride and padding enabled.
    model.add (MaxPooling2D (pool_size = (2,2), strides = (2,2), padding = 'valid'))
    # Batch Standardization
    model.add (BatchNormalization ())

    # We go to the Fully Connected (FC) stage
    model.add (Flatten ())
    # First layer Dense
    model.add (Dense (4096))
    model.add (Activation ('relu'))
    # Dropout is introduced with probability equal to 0.4 to avoid overfitting.
    model.add (Dropout (0.4))
    # Batch Standardization
    model.add (BatchNormalization ())

    # Second layer Dense
    model.add (Dense (4096))
    model.add (Activation ('relu'))
    # # Dropout is introduced
    model.add (Dropout (0.4))
    # Batch Standardization
    model.add (BatchNormalization ())

    # Third layer Dense
    model.add (Dense (1000))
    model.add (Activation ('relu'))
    # Dropout is introduced
    model.add (Dropout (0.4))
    # Batch Standardization
    model.add (BatchNormalization ())

    # Output layer with 3 nodes, since the classification will be made between 3 categories
    model.add (Dense (17))
    model.add (Dense (3, activation = 'softmax'))
    
    return model

#AlexNet model is created as defined above
model3 = create_AlexNet ()
model3.compile (loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

hist_model3=model3.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=20)

#RESULT AND CONCLUSIONS

#We represent the evolution of the Accuracy for the train and validation set of the three models:
    # -CNN simple
    # -CNN simple with Data Augmentation
    # -AlexNet
plt.title ('Accuracy')
plt.plot (hist_model1.history ['accuracy'], color = 'blue', label = 'Simple CNN train set')
plt.plot (hist_model1.history ['val_accuracy'], color = 'orange', label = 'Simple CNN validation set')
plt.plot (hist_model2.history ['accuracy'], color = 'red', label = 'Simple CCN with Data Augmentation train set')
plt.plot (hist_model2.history ['val_accuracy'], color = 'green', label = 'Simple CNN without Data Augmentation validation set')
plt.plot (hist_model3.history ['accuracy'], color = 'black', label = 'AlexNet train set')
plt.plot (hist_model3.history ['val_accuracy'], color = 'gray', label = 'AlexNet CNN validation set')

plt.legend (['Simple CNN train set', 'Simple CNN validation set',
 'Simple CCN with Data Augmentation train set', 'Validation set in CNN without Data Augmentation', 'AlexNet train set',
           'AlexNet CNN validation set'])
plt.rcParams ["figure.figsize"] = [30.10]
plt.show (1)

#Firstly, we evaluated the classifier obtained by training the Simple CNN model on the test set.
_, acc1 = model1.evaluate(x_test, y_test, verbose=0)
print('%.3f' % (acc1 * 100.0))

#We get the confusion matrix for the simple CNN model
labTrue = ["HEALTHY: current", "COVID-19: current", "PNEUMONIA: current"]
labPred = ["HEALTHY: pred", "COVID-19: pred", "PNEUMONIA: pred"]
y_pred1 = model1.predict_classes (x_test, verbose = 0)
y_test_real = np.argmax (y_test, axis = 1)
conf_matrix1 = confusion_matrix (y_test_real, y_pred1)
df_conf_matrix1 = pd.DataFrame (conf_matrix1, index = labTrue,
                   columns = labPred)
df_conf_matrix1

# Second, we evaluate the classifier obtained by training the CNN Simple model on the test set
#with Data Augmentation.
_, acc2 = model2.evaluate (x_test, y_test, verbose = 0)
print ('%. 3f'% (acc2 * 100.0))

#We get the confusion matrix for the simple CNN model with Data Augmentation
y_pred2 = model2.predict_classes (x_test, verbose = 0)
conf_matrix2 = confusion_matrix (y_test_real, y_pred2)
df_conf_matrix2 = pd.DataFrame (conf_matrix2, index = labTrue,
                   columns = labPred)
df_conf_matrix2

#Finally, the model is evaluated on the test set, the classifier obtained by training the model
#AlexNet.
_, acc3 = model3.evaluate (x_test, y_test, verbose = 0)
print ('%. 3f'% (acc3 * 100.0))

#We get the confusion matrix for the AlexNet model
y_pred3 = model3.predict_classes (x_test, verbose = 0)
conf_matrix3 = confusion_matrix (y_test_real, y_pred3)
df_conf_matrix3 = pd.DataFrame (conf_matrix3, index = labTrue,
                   columns = labPred)
df_conf_matrix3

#EXPLAINABILITY

#Below you can see the effect of the first convolutional layer on a particular image. Is a
# way of seeing how the AlexNet model is working on an image. This process could be applied for each
#one of the layers of the model and for each one of the images.

#We see that some filters focus on the bone area and most filters focus on the corresponding mass
#to the lungs.

#The influence of the convolutional layers on the following image will be analyzed.
img_CAM = load_img ('COVID-19 / 41591_2020_819_Fig1_HTML.webp-day10.png', target_size = (224, 224))
plt.imshow (img_CAM, cmap = 'gray')

# The first convolutional layer of the created AlexNet model is loaded into the model variable.
model = Model (inputs = model3.inputs, outputs = model3.layers [1] .output)
model.summary ()
# The image is loaded with the corresponding size
img_CAM = load_img ('COVID-19 / 41591_2020_819_Fig1_HTML.webp-day10.png', target_size = (224, 224))
# Convert image to array
img_CAM = img_to_array (img_CAM)
# Dimension expands
img_CAM = expand_dims (img_CAM, axis = 0)
# Pixels of the image are preprocessed according to the first convolutional layer
img_CAM = preprocess_input (img_CAM)
# The characteristics map is obtained after the first convolutional layer
feature_maps = model.predict (img_CAM)
# 64 samples are displayed in an 8x8 matrix
square = 8
ix = 1
for _ in range (square):
for _ in range (square):
ax = plt.subplot (square, square, ix)
ax.set_xticks ([])
ax.set_yticks ([])
plt.imshow (feature_maps [0,:,:, ix-1], cmap = 'gray', aspect = 'auto')
ix + = 1
plt.rcParams ["figure.figsize"] = [10.5]
plt.show ()

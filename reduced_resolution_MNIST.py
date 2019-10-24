# augmenting the data by reducing the resolution/ image pixel by decreasing 2 from 28 to 2 
# observe how the augmented data affects the accuracy of the trained model 

import os
import cv2
from tensorflow.keras.models import load_model
import tensorflow.keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
import random
import time
import pandas as pd
from matplotlib import pyplot
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import display
from numpy import genfromtxt
import pickle
from scipy import ndimage
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

#enter directory name
dict_test = "..."
CATEGORIES = ["0","1","2","3","4","5","6","7","8","9"]


IMG_SIZE = 2
img_size2 = 28 #reduce resolution

error = []
TEST = []
label = []
TEST2 = []
    
for category in CATEGORIES:
    path = os.path.join(dict_test,category)
    for img in os.listdir(path):
        #try:
        img_array = cv2.imread((os.path.join(path, img)), cv2.IMREAD_GRAYSCALE) # convert to array
        new_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
        new_array2 = cv2.resize(new_array, (img_size2,img_size2))
        TEST.append(new_array2)
        label.append(category)
        plt.imshow(new_array2, cmap='gray') 
        plt.show()  
        #except:
            #error.append(img)
            #error.append(category)
            #pass
#print(error)

TEST = np.array(TEST).reshape(-1,img_size2,img_size2,1)

print(np.shape(TEST))
#print(np.shape(label))
#print(label)
#print(TEST[:])
#print(TEST)
#print(y_test)
#print(TEST.shape)
#print(label.shape)
#print("Shape of x_test: {}".format(x_test.shape))
#print("Shape of y_test: {}".format(y_test.shape))
#print(label)

model = load_model("my_CNN_MNIST_model.h5")

model.summary()

label_ = []
for i in label:
    label_.append(int(i))
    
num_classes = 10

label = tensorflow.keras.utils.to_categorical(label_, num_classes)

# Predict using CPU, send the entire dataset.
# Set the desired TensorFlow output level for this example
loss, acc = model.evaluate(TEST, label)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))







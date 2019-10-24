#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


dict_test = "D:\\rotated_class_data\\rotated_150"
#dict_test = "C:\\Users\\Tan Ek Huat\\Desktop\\test_data"
CATEGORIES = ["0","1","2","3","4","5","6","7","8","9"]
#CATEGORIES = ["0"]

IMG_SIZE = 28

error = []
TEST = []
label = []

    
for category in CATEGORIES:
    path = os.path.join(dict_test,category)
    for img in os.listdir(path):
        try:
            img_array = cv2.imread((os.path.join(path, img)), cv2.IMREAD_GRAYSCALE) # convert to array
            new_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
            TEST.append(new_array)
            label.append(category)
        except:
            error.append(img)
            error.append(category)
            pass
print(error)

TEST = np.array(TEST).reshape(-1,IMG_SIZE,IMG_SIZE,1)


# In[3]:


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


# In[4]:


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


# In[ ]:





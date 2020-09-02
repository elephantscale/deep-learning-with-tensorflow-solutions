#!/usr/bin/env python

## This CNN network to identify MNIST dataset

import time
import random
import numpy as np
from pprint import pprint
import os, sys

## disable info logs from TF
#   Level | Level for Humans | Level Description                  
#  -------|------------------|------------------------------------ 
#   0     | DEBUG            | [Default] Print all messages       
#   1     | INFO             | Filter out INFO messages           
#   2     | WARNING          | Filter out INFO & WARNING messages 
#   3     | ERROR            | Filter out all messages 

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}

import tensorflow as tf
tf.get_logger().setLevel('WARN')
from tensorflow import keras
print (tf.config.experimental.list_physical_devices())

## ---- start Memory setting ----
## Ask TF not to allocate all GPU memory at once.. allocate as needed
## Without this the execution will fail with "failed to initialize algorithm" error

from tensorflow.compat.v1.keras.backend import set_session
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
set_session(sess)
## ---- end Memory setting ----


# ## Step 1: Download Data

mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
print()
print("~~~~~~ here is train/test size:")
print("train_images shape : ", train_images.shape)
print("train_labels shape : ", train_labels.shape)
print("test_images shape : ", test_images.shape)
print("test_labels shape : ", test_labels.shape)


# ## Step 2 : Shape Data

# ### 2.1 - Shape the array to 4 dimensional
# ConvNets expect data in 4D.  Let's add a channel dimension to our data.

## Reshape to add 'channel'.
train_images = train_images.reshape(( train_images.shape[0],  28, 28, 1))
test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))

# print("train_images shape : ", train_images.shape)
# print("train_labels shape : ", train_labels.shape)
# print("test_images shape : ", test_images.shape)
# print("test_labels shape : ", test_labels.shape)


# ### 2.2 - Normalize Data
# The images are stored as a 2D array of pixels.  
# Each pixel is a value from 0 to 255  
# We are going to normalize them in the range of 0 to 1
## Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0




# ## Step 3 : Create Model
# We are definiing a LeNet architecture
# Reference  : https://www.pyimagesearch.com/2016/08/01/lenet-convolutional-neural-network-in-python/

# ### 3.1 - Create a CNN

# As input, a CNN takes tensors of shape (image_height, image_width, color_channels), ignoring the batch size. If you are new to color channels, MNIST has one (because the images are grayscale), whereas a color image has three (R,G,B). In this example, we will configure our CNN to process inputs of shape (28, 28, 1), which is the format of MNIST images. We do this by passing the argument `input_shape` to our first layer.

model = tf.keras.Sequential( [ 
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
])

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

print()
print("~~~~~ Model summary:")
print(model.summary())


# ### 3.2 - Compile and Train

model.compile(optimizer=tf.keras.optimizers.Adam(),  # 'adam'
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


## Step 4 : Train

epochs = 10
print()
print ("~~~~~ training starting ...")
t1 = time.perf_counter()
history = model.fit(train_images, train_labels, 
                    epochs=epochs, validation_split = 0.2, verbose=1)
t2 = time.perf_counter()
print ("~~~~~ trained on {:,} images in {:,.2f} ms".format (len(train_images), (t2-t1)*1000))


# ## Step 5 : Predict


# these predictions will be softmax arrays with probabilities
predictions = model.predict(test_images)
# ## our predictions is an array of arrays
# print('predictions shape : ', predictions.shape)
# print ('prediction 0 : ' , predictions[0])
# print ('prediction 1 : ' , predictions[1])

## predict actual labels (0,1,2 ...)
t1 = time.perf_counter()
predictions2 = model.predict_classes(test_images)
t2 = time.perf_counter()
print()
print ("~~~~~ predicted {:,} images in {:,.2f} ms".format (len(test_images), (t2-t1)*1000))



# ## Step 6 : Evaluate the Model 
# ### 6.1 - Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_labels, predictions2, labels = [0,1,2,3,4,5,6,7,8,9])

print ('~~~~~ Here is the confusion matrix:')
print(cm)


# ### 6.2 - Metrics
metric_names = model.metrics_names
print()
print ("~~~~ model metrics : ")
metrics = model.evaluate(test_images, test_labels, verbose=0)

for idx, metric in enumerate(metric_names):
    print ("    Metric : {} = {:,.3f}".format (metric_names[idx], metrics[idx]))




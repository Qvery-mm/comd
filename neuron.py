
# coding: utf-8

# In[2]:

from keras import optimizers, losses, metrics #applications,
#from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model #Sequential,
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, Input, Convolution2D, MaxPooling2D, Conv2D, Conv1D, Activation
#from keras import backend as k
#from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
#from keras.datasets import cifar10 # subroutines for fetching the CIFAR-10 dataset
#from keras.utils import np_utils # utilities for one-hot encoding of ground truth values
import numpy as np
#from keras import backend as K
from PIL import Image
#from sklearn import metrics as sklearn_metrics
import cv2
import os



# In[3]:

batch_size = 4
kernel_size = 3
pool_size = 2
conv_depth_1 = 64
conv_depth_2 = 128
drop_prob_1 = 0.25
drop_prob_2 = 0.5
hidden_size = 128
strides = 3


# In[4]:

sample_weight = 1
depth = 1
img_width, img_height = 512, 512
inp = Input(shape=( depth, img_height, img_width))
conv_1 = Convolution2D(conv_depth_1, kernel_size, strides = 1, border_mode='same', activation='relu', data_format='channels_first')(inp)
pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size),data_format='channels_first')(conv_1)
drop_1 = Dropout(drop_prob_1)(pool_1)
conv_2 = Convolution2D(conv_depth_2, kernel_size, strides = 1, border_mode='same', activation='relu', data_format='channels_first')(drop_1)
pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size),data_format='channels_first')(conv_2)
drop_2 = Dropout(drop_prob_1)(pool_2)
flat = Flatten()(drop_2)
hidden = Dense(hidden_size, activation='relu')(flat)
drop_3 = Dropout(drop_prob_2)(hidden)
hidden_2 = Dense(hidden_size, activation='relu')(drop_3)
drop_4 = Dropout(drop_prob_2)(hidden_2)
out = Dense(1, activation = "sigmoid")(drop_3)


# In[5]:

model = Model(input = inp, output = out)


# In[6]:

sgd = optimizers.SGD(lr=0.01)
model.compile(loss= losses.binary_crossentropy,
              optimizer= sgd,
              metrics=[metrics.mae, 'accuracy'])
model.load_weights("weights.best.bezvgg19_6.hdf5")

print("finish initialising")
# In[71]:

def kaskad(image):
#     image = cv2.imread("Image/legkoe.png", 0)
    cascadePath = "HAAR.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)


    faces = faceCascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))

    images = []
    labels = []
#     return len(faces)
    for (x, y, w, h) in faces:
        #cv2.imshow("", image[y: y + h, x: x + w])
        image = image[y: y + h+40, x: x + w]
        return image


# In[88]:

def kaskad_2(image):
    #image = cv2.imread("Neuron/legkoe.png", 0)
    cascadePath = "HAAR.xml"
    faceCascade = cv2.CascadeClassifier(cascadePath)


    faces = faceCascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))

    images = []
    labels = []

    for (x, y, w, h) in faces:
          #cv2.imshow("", image[y: y + h, x: x + w])
        image = image[y: y + h+40, x: x + w]
    return image
        #cv2.imwrite("Neuron/legkoe.png",  image)


# In[95]:

def obrezka(im):
#     im = im.convert("L")
    img = kaskad_2(im)
#     print(im)

    img = cv2.resize(img, (512, 512))
    #img = img.convert("L")
    return(img)


# In[93]:

def predict(t):
#     im.show()
    im = cv2.imread(t, 0)
    im = obrezka(im)
    im_testing = np.array([np.array([np.array(im)])])
    pred = model.predict([im_testing], batch_size=4, verbose=0)
    return pred[0][0]


# In[101]:





from __future__ import print_function
import os
import datetime
import itertools
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, AveragePooling2D, Dropout, \
    BatchNormalization
from keras.optimizers import Adam
from keras.layers.merge import Add, Concatenate
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.layers.advanced_activations import LeakyReLU
import cv2
 
PIXEL = 512    #set your image size
BATCH_SIZE = 5
lr = 0.001
EPOCH = 100
X_CHANNEL = 3  # training images channel
Y_CHANNEL = 3  # label iamges channel
X_NUM = 100  # your traning data number
 
pathX = './database/image/'    #change your file path
pathY = './database/mask/'    #change your file path
 
def generator(pathX, pathY,BATCH_SIZE):
    while 1:
        X_train_files = os.listdir(pathX)
        Y_train_files = os.listdir(pathY)
        a = (np.arange(1, X_NUM))
        X = []
        Y = []
        for i in range(BATCH_SIZE):
            index = np.random.choice(a)
            # print(index)
            img = cv2.imread(pathX + X_train_files[index], 1)
            img = np.array(img).reshape(PIXEL, PIXEL, X_CHANNEL)
            X.append(img)
            img1 = cv2.imread(pathY + Y_train_files[index], 1)
            img1 = np.array(img1).reshape(PIXEL, PIXEL, Y_CHANNEL)
            Y.append(img1)
 
        X = np.array(X)
        Y = np.array(Y)
        yield X, Y
 
 #creat unet network
inputs = Input((PIXEL, PIXEL, 3))

#Convolution Block1 (CB1)
conv1 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
pool1 = AveragePooling2D(pool_size=(2, 2))(conv1)

#Convolution Block2 (CB2)
conv2 = BatchNormalization(momentum=0.99)(pool1)
conv2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
conv2 = BatchNormalization(momentum=0.99)(conv2)
conv2 = Conv2D(64, 1, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
conv2 = Dropout(0.02)(conv2)
pool2 = AveragePooling2D(pool_size=(2, 2))(conv2)  # 8

#Convolution Block3 (CB3)
conv3 = BatchNormalization(momentum=0.99)(pool2)
conv3_1 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
conv3 = BatchNormalization(momentum=0.99)(conv3_1)
conv3_2 = Conv2D(128, 1, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
conv3 = Dropout(0.02)(conv3_2)
pool3 = AveragePooling2D(pool_size=(2, 2))(conv3)  # 4

#Dliated Block1 (DB1)
conv4 = BatchNormalization(momentum=0.99)(pool3)
conv4_1 = Conv2D(256, 3, activation='relu', padding='same', dilation_rate=(2, 2), kernel_initializer='he_normal')(conv4)
conv4 = BatchNormalization(momentum=0.99)(conv4_1)
conv4_2 = Conv2D(256, 1, activation='relu', padding='same', dilation_rate=(2, 2), kernel_initializer='he_normal')(conv4)
conv4 = Add()([conv4_1, conv4_2]) #Feature reuse module
conv4 = Dropout(0.02)(conv4)

#Dliated Block2 (DB2)
conv5 = BatchNormalization(momentum=0.99)(conv4)
conv5_1 = Conv2D(512, 3, activation='relu', padding='same', dilation_rate=(4, 4), kernel_initializer='he_normal')(conv5)
conv5 = BatchNormalization(momentum=0.99)(conv5_1)
conv5_2 = Conv2D(512, 1, activation='relu', padding='same', dilation_rate=(4, 4), kernel_initializer='he_normal')(conv5)
conv5 = Add()([conv5_1, conv5_2]) #Feature reuse module
conv5 = Concatenate()([pool3, conv4, conv5]) #Feature fusion module
conv5 = Conv2D(512, 1, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
conv5 = Dropout(0.02)(conv5)
 
conv6 = BatchNormalization(momentum=0.99)(conv5)
conv6 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

#Convolution Block4 (CB4)
conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
merge7 = concatenate([conv4, conv7], axis=3) #skip connection

#Convolution Block5 (CB5)
conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
merge8 = concatenate([pool3, conv8], axis=3) #skip connection

#Upsample (UP1)
conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
up9 = (UpSampling2D(size=(2, 2))(conv9))
conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up9)
merge9 = concatenate([pool2, conv9], axis=3) #skip connection

#Upsample (UP2)
conv10 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
up10 = (UpSampling2D(size=(2, 2))(conv10))
conv10 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up10)

#Upsample (UP3)
conv11 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv10)
up11 = (UpSampling2D(size=(2, 2))(conv11)) 
conv11 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(up11)

conv12 = Conv2D(3, 1, activation='relu', padding='same', kernel_initializer='he_normal')(conv11)
 
model = Model(input=inputs, output=conv12)
print(model.summary())
model.compile(optimizer=Adam(lr=1e-3), loss='mse', metrics=['accuracy'])
 
history = model.fit_generator(generator(pathX, pathY,BATCH_SIZE),
                              steps_per_epoch=20, nb_epoch=EPOCH)
end_time = datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S')
 
 #save your training model
model.save(r'V1_828.h5')
 
#save your loss data
mse = np.array((history.history['loss']))
np.save(r'V1_828.npy', mse)



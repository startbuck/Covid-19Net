from __future__ import print_function
import os
import datetime
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, AveragePooling2D, Dropout, BatchNormalization, Reshape
from keras.optimizers import Adam
from keras.layers.core import Activation
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.layers.advanced_activations import LeakyReLU
import cv2
from keras.layers.merge import Add, Concatenate
 
PIXEL = 512    #set your image size
BATCH_SIZE = 5
lr = 0.001
EPOCH = 1
X_CHANNEL = 3  # training images channel
Y_CHANNEL = 3  # label iamges channel
X_NUM = 100  # your traning data number
 
pathX = './database/image/'    #change your file path
pathY = './database/mask/'    #change your file path
 
#data processing
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
 
conv1 = Conv2D(filters=32, input_shape=(PIXEL, PIXEL, 3),
                kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal',
                name='block1_conv1')(inputs)
conv1 = Conv2D(filters=32, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal',
                   name='block1_conv2')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2), name='block1_pool')(conv1)
 
conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal',
                   name='block2_conv1')(pool1)
conv2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal',
                   name='block2_conv2')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2), name='block2_pool')(conv2)
 
 
conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal',
                   name='block3_conv1')(pool2)
conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal',
                   name='block3_conv2')(conv3)
conv3 = Conv2D(filters=128, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal',
                   name='block3_conv3')(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2), name='block3_pool')(conv3)
    #8
score_pool3 = Conv2D(filters=3, kernel_size=(3, 3),padding='same',
                         activation='relu', kernel_initializer='he_normal',name='score_pool3')(pool3)
    
 
conv4 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal',
                   name='block4_conv1')(pool3)
conv4 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal',
                   name='block4_conv2')(conv4)
conv4 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal',
                   name='block4_conv3')(conv4)
pool4 = MaxPooling2D(pool_size=(2, 2), name='block4_pool')(conv4)
    #16
score_pool4 = Conv2D(filters=3, kernel_size=(3, 3),padding='same',
                         activation='relu', kernel_initializer='he_normal',name='score_pool4')(pool4)
    
 
conv5 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal',
                   name='block5_conv1')(pool4)
conv5 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal',
                   name='block5_conv2')(conv5)
conv5 = Conv2D(filters=256, kernel_size=(3, 3), padding='same', activation='relu', kernel_initializer='he_normal',
                   name='block5_conv3')(conv5)
pool5 = MaxPooling2D(pool_size=(2, 2), name='block5_pool')(conv5)
    
    
    
fc6 = Conv2D(filters=256, kernel_size=(1, 1), padding='same', activation='relu', kernel_initializer='he_normal',
                 name='fc6')(pool5)
fc6 = Dropout(0.3, name='dropout_1')(fc6)
 
fc7 = Conv2D(filters=256, kernel_size=(1, 1), padding='same', activation='relu', kernel_initializer='he_normal',
                 name='fc7')(fc6)
fc7 = Dropout(0.3, name='dropour_2')(fc7)
    
    
    #32
score_fr = Conv2D(filters=3, kernel_size=(1, 1), padding='same', activation='relu',kernel_initializer='he_normal',name='score_fr')(fc7)
    #Conv2DTranspose
score2 = Conv2DTranspose(filters=3, kernel_size=(2, 2), strides=(2, 2),
                        padding="valid", activation=None,
                        name="score2")(score_fr)
    
    #add 32 and 16
add1 = Add()([score2,score_pool4])
 
score4 = Conv2DTranspose(filters=3, kernel_size=(2, 2), strides=(2, 2),
                             padding="valid", activation=None,
                             name="score4")(add1)
    #add the sum of the 32 and 16 and 8
add2 = Add()([score4,score_pool3])
    #use transpose convolution to recover the resolution of the input
UpSample = Conv2DTranspose(filters=3, kernel_size=(8, 8), strides=(8, 8),
                             padding="valid", activation=None,
                             name="UpSample")(add2)
UpSample = Activation("softmax")(UpSample)
model = Model(input=inputs, output=UpSample)

model.compile(optimizer=Adam(lr=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
 
history = model.fit_generator(generator(pathX, pathY,BATCH_SIZE),
                              steps_per_epoch=20, nb_epoch=EPOCH)
end_time = datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S')
 
 #save your training model
model.save(r'V1_828.h5')
 
#save your loss data
mse = np.array((history.history['loss']))
np.save(r'V1_828.npy', mse)



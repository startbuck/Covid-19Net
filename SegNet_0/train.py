from __future__ import print_function
import os
import datetime
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, AveragePooling2D, Dropout, \
    BatchNormalization, Reshape, Permute
from keras.optimizers import Adam
from keras.layers.core import Activation
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.layers.advanced_activations import LeakyReLU
import cv2
from keras.layers.merge import Add, Concatenate
 
PIXEL = 512    #set your image size
BATCH_SIZE = 1
lr = 0.001
EPOCH = 100
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

    # encoder
inputs = Input((PIXEL, PIXEL, 3))

conv1 = Conv2D(filters=64,input_shape=(512, 512, 3),kernel_size=(3,3),strides=(1,1),padding='same',activation='relu')(inputs)
conv1 = Conv2D(filters=64,kernel_size=(3,3),strides=(1,1),padding='same',activation='relu')(conv1)
conv1 = BatchNormalization()(conv1)
conv1 = MaxPooling2D(pool_size=(2,2))(conv1)
    #(128,128)
conv2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv1)
conv2 = BatchNormalization()(conv2)
conv2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
conv2 = BatchNormalization()(conv2)
conv2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    #(64,64)
conv3 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv2)
conv3 = BatchNormalization()(conv3)
conv3 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3)
conv3 = BatchNormalization()(conv3)
conv3 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3)
conv3 = BatchNormalization()(conv3)
conv3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    #(32,32)
conv4 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv3)
conv4 = BatchNormalization()(conv4)
conv4 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv4)
conv4 = BatchNormalization()(conv4)
conv4 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv4)
conv4 = BatchNormalization()(conv4)
conv4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    #(16,16)
conv5 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv4)
conv5 = BatchNormalization()(conv5)
conv5 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv5)
conv5 = BatchNormalization()(conv5)
conv5 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv5)
conv5 = BatchNormalization()(conv5)
conv5 = MaxPooling2D(pool_size=(2, 2))(conv5)
    #(8,8)
    #decoder
conv5 = UpSampling2D(size=(2,2))(conv5)
    #(16,16)
conv6 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv5)
conv6 = BatchNormalization()(conv6)
conv6 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv6)
conv6 = BatchNormalization()(conv6)
conv6 = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv6)
conv6 = BatchNormalization()(conv6)
conv6 = UpSampling2D(size=(2, 2))(conv6)
    #(32,32)
conv7 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv6)
conv7 = BatchNormalization()(conv7)
conv7 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv7)
conv7 = BatchNormalization()(conv7)
conv7 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv7)
conv7 = BatchNormalization()(conv7)
conv7 = UpSampling2D(size=(2, 2))(conv7)
    #(64,64)
conv8 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv7)
conv8 = BatchNormalization()(conv8)
conv8 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv8)
conv8 = BatchNormalization()(conv8)
conv8 = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv8)
conv8 = BatchNormalization()(conv8)
conv8 = UpSampling2D(size=(2, 2))(conv8)
    #(128,128)
conv9 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv8)
conv9 = BatchNormalization()(conv9)
conv9 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv9)
conv9 = BatchNormalization()(conv9)
conv9 = UpSampling2D(size=(2, 2))(conv9)
    #(256,256)
conv9 = Conv2D(filters=64,input_shape=(512, 512, 3), kernel_size=(3, 3), strides=(1, 1),  padding='same', activation='relu')(conv9)
conv9 = BatchNormalization()(conv9)
conv9 = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(conv9)
conv9 = BatchNormalization()(conv9)
conv9 = Conv2D(3, (1, 1), strides=(1, 1), padding='same')(conv9)

conv9 = Activation("softmax")(conv9)

model = Model(inputs=inputs, outputs=conv9, name="SegNet")
model.compile(loss='categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])
model.summary()
 
history = model.fit_generator(generator(pathX, pathY,BATCH_SIZE),
                              steps_per_epoch=100, nb_epoch=EPOCH)
end_time = datetime.datetime.now().strftime('%Y-%m-%d  %H:%M:%S')
 
 #save your training model
model.save(r'V1_828.h5')
 
#save your loss data
mse = np.array((history.history['loss']))
np.save(r'V1_828.npy', mse)



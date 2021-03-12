from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
 
model = load_model('V1_828.h5')
test_images_path = './database/test/image/'
test_gt_path = './database/test/groundtruth/'
pre_path = './database/test/image/predict/'
 
X = []
for info in os.listdir(test_images_path):
    A = cv2.imread(test_images_path + info)
    X.append(A)
    # i += 1
X = np.array(X)
print(X.shape)
Y = model.predict(X)
 
 
groudtruth = []
for info in os.listdir(test_gt_path):
    A = cv2.imread(test_gt_path + info)
    groudtruth.append(A)
groudtruth = np.array(groudtruth)
 
i = 0
for info in os.listdir(test_images_path):
    cv2.imwrite(pre_path + info,Y[i])
    i += 1
 
 
test_loss_acc = model.evaluate(X, groudtruth)
print(test_loss_acc)
for n in range(10):
	cv2.imwrite(str(n)+'prediction.png',Y[n])
	cv2.imwrite(str(n)+'groudtruth.png',groudtruth[n])



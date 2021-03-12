from keras.models import load_model
import numpy as np
import matplotlib.patches as mpatches
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
 
 
a = range(10)
n = np.random.choice(a)
Y_lbls = np.argmax(Y[n],axis=-1)

all_labels = ['ground class opacification', 'consolidations', 'pleural effusions']
colors = np.asarray([[141,211,199,255],
                     [255,255,179,255],
                     [179,222,105,255]]) / 255.

plt.subplot(1, 2, 1)
plt.imshow(np.squeeze(X[n]),cmap='gray')
plt.subplot(1, 2, 2)
plt.imshow(colors[np.squeeze(Y_lbls)])
plt.legend(handles=[mpatches.Patch(color=colors[i], label=all_labels[i]) for i in np.unique(Y_lbls)], prop={'size':6})
plt.show()



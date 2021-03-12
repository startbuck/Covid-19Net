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
 
 
a = range(1)
n = np.random.choice(a)
test_loss_acc = model.evaluate(X, groudtruth)
print(test_loss_acc)
cv2.imwrite('prediction.png',Y[n])
cv2.imwrite('groudtruth.png',groudtruth[n])
fig, axs = plt.subplots(1, 3)
# cnt = 1
# for j in range(1):
axs[0].imshow(np.abs(X[n]))
axs[0].axis('off')
axs[1].imshow(np.abs(Y[n]))
axs[1].axis('off')
axs[2].imshow(np.abs(groudtruth[n]))
axs[2].axis('off')
    # cnt += 1
fig.savefig("imagestest.png")
plt.close()



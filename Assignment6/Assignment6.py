import cv2
import numpy as np
from matplotlib import pyplot as plt

'''
Otsu’s Thresholding: Apply Otsu’s thresholding tothe image ‘noisy_leaf.jpg’ toobtain the optimal threshold value to separate the foreground and backgroundeffectively as shown below.  (Apply gaussian blur before thresholding). Save thethresholded image.
'''

# Code
img = cv2.imread('noisy_leaf.jpg', cv2.IMREAD_GRAYSCALE)
blur = cv2.GaussianBlur(img, (5, 5), 0)
ret2, th2 = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imwrite('Otsus.jpg', th2)

'''
Adaptive thresholding: The image ‘page.jpg’ has illuminationvariation across it.Apply both ‘global’ and ‘local’ thresholding and compare and save the resultingimages.
'''

# Code
img = cv2.imread('page.jpg', cv2.IMREAD_GRAYSCALE)

blur = cv2.GaussianBlur(img, (5, 5), 0)

ret, th1 = cv2.threshold(blur, 150, 255, cv2.THRESH_BINARY)
cv2.imwrite('GlobalThres.jpg', th1)

th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
cv2.imwrite('AdaptiveThres.jpg', th3)

'''
Segmentation with Grabcut: OpenCV samples (../OpenCV/samples/python/)contain a samplegrabcut.pywhich is an interactivetool using grabcut. Use it tosegment Messi out of the image ‘messi.jpg’ and save the result.youtube video on using the tool:https://www.youtube.com/watch?v=kAwxLTDDAwU
'''

# Code
import numpy as np
import cv2

img = cv2.imread('messi.jpg')
cv2.imshow('img', img)

mask = np.zeros(img.shape[:2], np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)
rect = (70, 65, 400, 290)

cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
img_cut = img * mask2[:, :, np.newaxis]

cv2.imwrite('grapcut_messi.jpg', img_cut)

'''
K-means clustering for color quantization:Apply k-meansclustering to the image‘building.png’ to cluster different color defined regions together. Here the value of ‘k’determines the number of colors to which the image is quantised. Save the results ofclustering with k=3, 5 and 8.
'''

# Code

import numpy as np
import cv2

img = cv2.imread('building.png')
Z = img.reshape((-1, 3))

Z = np.float32(Z)


def k_means_clustering(k, ):
    ret, label, center = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    cv2.imwrite(f'{k}meanCluster.jpg', res2)


criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

k_means_clustering(3)
k_means_clustering(5)
k_means_clustering(8)

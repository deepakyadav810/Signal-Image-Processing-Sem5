import cv2 
import numpy as np


#1
image = cv2.imread('Lenna.png')
cv2.imshow('Original', image)
cv2.waitKey(0)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Grayscale', gray_image)
cv2.waitKey(0)
cv2.imwrite(r"GrayLenna.png",gray_image)

#criteria
img=cv2.imread('GrayLenna.png')
Z=img.reshape((-1,3))
Z = np.float32(Z)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

#quantize level 2
K = 2
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))
cv2.imshow('res2',res2)
cv2.imwrite("Lenna2.png",res2)
cv2.waitKey(0)

#quantize level 4
K = 4
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))
cv2.imshow('res2',res2)
cv2.imwrite("Lenna4.png",res2)
cv2.waitKey(0)

#quantize level 8
K = 8
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))
cv2.imshow('res2',res2)
cv2.imwrite("Lenna8.png",res2)
cv2.waitKey(0)

#quantize level 16
K = 16
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))
cv2.imshow('res2',res2)
cv2.imwrite("Lenna16.png",res2)
cv2.waitKey(0)

#quantize level 32
K = 32
ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
res2 = res.reshape((img.shape))
cv2.imshow('res2',res2)
cv2.imwrite("Lenna32.png",res2)
cv2.waitKey(0)

#2
image_1 = cv2.imread('aimg1.jpg')
image_2 = cv2.imread('aimg2.jpg')

#2.a)
img = image_1+image_2
cv2.imshow('pixle_by_pixel_add',img)
cv2.imwrite("pixel_by_pixel.jpg", img)
cv2.waitKey(0)

#2.b)
addImage = cv2.add(image_1, image_2)
cv2.imshow("add image", addImage)
cv2.imwrite("add.jpg", addImage)
cv2.waitKey(0)

#2.c)
weightedSum = cv2.addWeighted(image_1, 0.5, image_2, 0.5, 0)
cv2.imshow('Weighted Image', weightedSum)
cv2.imwrite("Weighted_Image.jpg", weightedSum)
cv2.waitKey(0)

#3
simg1 = cv2.imread('simg1.png') 
simg2 = cv2.imread('simg2.png')

simg1_gray = cv2.cvtColor(simg1, cv2.COLOR_BGR2GRAY)
simg2_gray = cv2.cvtColor(simg2, cv2.COLOR_BGR2GRAY)

sub = cv2.subtract(simg1_gray, simg2_gray)

ret, thresh = cv2.threshold(sub, 0, 255, cv2.THRESH_BINARY)
cv2.imshow('Subtracted Image', thresh)
cv2.imwrite("Subtracted_Image.jpg", thresh)    
cv2.waitKey(0) 

#4
teeth = cv2.imread('teeth.jpg') 
mask = cv2.imread('mask.jpg')

teeth_gray = cv2.cvtColor(teeth, cv2.COLOR_BGR2GRAY)
mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

masked = cv2.bitwise_and(mask_gray, teeth_gray, mask=None)
cv2.imshow("Mask Applied to Image", masked)
cv2.imwrite("MaskedImage.jpg", masked) 
cv2.waitKey(0)

#5
board = cv2.imread(r'ChessBoardGrad.png').astype(np.float32)
shade = cv2.imread(r'shading.png').astype(np.float32)

board_gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
shade_gray = cv2.cvtColor(shade, cv2.COLOR_BGR2GRAY)

board_div = cv2.divide(board_gray, shade_gray)

board_normalise = cv2.normalize(board_div, None, 0, 255, cv2.NORM_MINMAX)
cv2.imshow('imgNormalize1', board_normalise.astype(np.uint8))
cv2.imwrite("MaskedImage1.jpg", board_normalise) 
cv2.waitKey(0)

cv2.destroyAllWindows() 

import cv2
import numpy as np
from matplotlib import pyplot as plt

'''
Q1) Estimate the transformation function required for transforming the image ‘inputq1.jpeg’
to match the image ‘transformed.jpeg’.
    a) Plot/Draw the transformation function
    b) Apply the transformation function and store the resulting image as ‘outputq1.jpeg’

'''

import cv2
import numpy as np

def piecewise(img,h,w):
    for i in range(h):
        for j in range(w):
            if(img[i][j] > 105 and img[i][j] < 165):
                img[i][j] =10


img = cv2.imread('inputq1.jpeg', 0)
(h,w) = img.shape[:2]

piecewise(img,h,w)
cv2.imwrite("outputq1.jpeg", img)

'''
Q2) 
Consider the input image: ‘logndlinear.jpg’
    a) The general form of the log transformation is s = c log( 1 + r ). Apply this
        transformation to the input image such that
        C = 255/(log (1 + m)), where m is the maximum pixel value in the image
        Store the result as ‘logq2.jpg’
    b) Apply the following transformation function to the input image
'''

#a

img = cv2.imread('logndlinear.jpg')  
c = 255/(np.log(1 + np.max(img))) 
log_transformed = c * np.log(1 + img)  
log_transformed = np.array(log_transformed, dtype = np.uint8) 
  
cv2.imwrite('log_transformed.jpg', log_transformed)

#b

def transformation(pix, r1, s1, r2, s2): 
    if (0 <= pix and pix <= r1): 
        return (s1 / r1)*pix 
    elif (r1 < pix and pix <= r2): 
        return ((s2 - s1)/(r2 - r1)) * (pix - r1) + s1 
    else: 
        return ((255 - s2)/(255 - r2)) * (pix - r2) + s2 
  

img = cv2.imread('logndlinear.jpg') 
r1 = 70
s1 = 0
r2 = 140
s2 = 255

pixelVal = np.vectorize(transformation) 
   
log_trans = pixelVal(img, r1, s1, r2, s2) 
 
cv2.imwrite('log_trans.jpg', log_trans) 

'''
Q4)
Consider the input image: ‘lowContrast.png’
    a) Plot its histogram and save the plot
    b) Perform histogram equalisation and save the equalised image
    c) Plot the equalised histogram and save the plot

'''

img = cv2.imread('lowContrast.png',0)
plt.hist(img.ravel(),256,[0,256]) 

plt.show() 
plt.savefig('hist.png')

equ = cv2.equalizeHist(img)
res = np.hstack((img,equ))

cv2.imshow('Equalized Image',res)
cv2.imwrite('Equalized Image.png',res)

plt.hist(res.ravel(),256,[0,256]) 

plt.show() 
plt.savefig('equal-hist.png')

'''
Q5) 
Blur the input image ‘building.png’ to three very distinct levels to result in images that
look like images ‘blurred_1.jpg’, ‘blurred_2.jpg’ and ‘blurred_3.jpg’.
'''

img = cv2.imread('building.png')

blur = cv2.blur(img,(10,10))
cv2.imwrite('blurredOut1.png', blur)

blur = cv2.blur(img,(30,30))
cv2.imwrite('blurredOut2.png', blur)


blur = cv2.blur(img,(50,50))
cv2.imwrite('blurredOut3.png', blur)

'''
Q6) 

Estimate the shading pattern in the image ‘ChessBoardGrad.png’. Store the estimate of
shading error as ‘shading.png’ and use this for shading correction. Store the corrected image
as ‘corrected.png’
'''

img = cv2.imread('ChessBoardGrad.png')
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
filtersize = 513
gaussianImg = cv2.GaussianBlur(grayImg, (filtersize, filtersize), 128)
cv2.imwrite('shading.png', gaussianImg)
newImg = (grayImg-gaussianImg)
cv2.imwrite('Corrected.png', newImg)

'''
Q7. Consider the input image: 1200px-Monarch_In_May.jpg. Convert this image to grayscale
and apply the following transformations:
a) Laplacian
b) Laplacian of Gaussian ( gaussian filter of size 3x3)
Save and compare the resulting images. Comment on the differences
'''

#a
laplacian = np.array(([0, 1, 0],[1, -4, 1],[0, 1, 0]), dtype="int")

image = cv2.imread("1200px-Monarch_In_May.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

opencvOutput = cv2.filter2D(gray, -1, laplacian)
cv2.imwrite("laplacianMonarch.jpg", opencvOutput)

#b
image = cv2.imread("1200px-Monarch_In_May.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(3,3),0)
opencvOutput = cv2.filter2D(gray, -1, laplacian)
cv2.imwrite("laplacianAndGauss.jpg", opencvOutput)

# A Laplacian filter is an edge detector which computes the second derivatives of an image, 
# measuring the rate at which the first derivatives change. That determines if a change in 
# adjacent pixel values is from an edge or continuous progression. Laplacian is very sensitive to 
# noise. It even detects the edges for the noise in the image. Laplacian kernel is very sensitive 
# to noise. Hence we use the Gaussian Filter to first smoothen the image and remove the noise. And 
# then the Laplacian Filter is applied for better results.


'''
Q8. Consider the input image: ‘ChessBoardGrad.png’.
Apply the following to the image:
a) Laplacian kernel
b) Sobel kernel in x direction
c) Sobel kernel in y direction
d) Canny edge detection
Save and compare the resulting images. Comment on the differences

'''

#a
image = cv2.imread("ChessBoardGrad.png")
grayLap = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
graySobelX = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
graySobelY = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#laplacian
imageLaplacian = cv2.filter2D(grayLap, -1, laplacian)
cv2.imwrite("laplacianChess.jpg", imageLaplacian)

#b
#SobelX
sobelX = np.array(([-1, 0, 1],[-2, 0, 2],[-1, 0, 1]), dtype="int")
imageSobelX = cv2.filter2D(gray, -1, sobelX)
cv2.imwrite("SobelXChess.jpg", imageSobelX)

#SobelY
sobelY = np.array(([-1, -2, -1],[0, 0, 0],[1, 2, 1]), dtype="int")
imageSobelY = cv2.filter2D(gray, -1, sobelY)
cv2.imwrite("SobelYChess.jpg", imageSobelY)

#c
img = cv2.imread('ChessBoardGrad.png',0)
edges = cv2.Canny(img,100,200)
cv2.imwrite('CannyChess.jpg', edges)

# The Sobel filter is used for edge detection. It works by calculating the gradient of image 
# intensity at each pixel within the image. Because of the second-order derivatives in Laplacian, 
# this gradient operator is more sensitive to noise than first-order gradient operators. Also, 
# the thresholded magnitude of the Laplacian operator produces double edges. 
# Canny Edge Detection is a multi-stage algorithm consisting of Noise reduciton, intensity gradient, 
# non maximum suppression, and thresholding

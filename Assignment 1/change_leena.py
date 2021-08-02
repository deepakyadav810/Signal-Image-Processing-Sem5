import cv2
from PIL import Image 
import PIL 

#1.a)
image = cv2.imread('Lenna.png')
cv2.imshow('Original', image)
cv2.waitKey(0)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow('Grayscale', gray_image)
cv2.waitKey(0)

cv2.imwrite(r"grayLenna.jpg",gray_image)

"""
1.b).a)-256 grayscale shades can be represented in this image
1.b).b)-It depend on each pixel,  If each pixel is given a value consisting of 8 bits per pixel,  we can represent 2^8=256 different colors or shades of gray.: 
"""
#1.c)
im1 = Image.open(r"grayLenna.jpg") 
  
# quantize a image 
im1 = im1.quantize(4)
imgGray = im1.convert('L')
imgGray.save('grayLenna4.jpg')
im1.show() 

#2
img = cv2.imread('calvinHobbes.jpeg')
  
print('Original Dimensions : ',img.shape)
 
width = 450
height = 450
dim = (width, height)
print("worked till here")
# resize image
resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
 
print('Resized Dimensions : ',resized.shape)
 
cv2.imshow("Resized image", resized)
cv2.imwrite("chScaled.jpg",resized)
cv2.waitKey(0)



cv2.destroyAllWindows()

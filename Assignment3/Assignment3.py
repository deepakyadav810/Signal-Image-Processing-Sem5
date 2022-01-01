import cv2
import numpy as np
import matplotlib.pyplot as plt

#1.a)
def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat

image = cv2.imread('calvinHobbes.jpeg')
cv2.imshow('Original', image)
cv2.waitKey(0)
angle=30
new_image1=rotate_image(image,angle)
cv2.imshow('New image1', new_image1)
cv2.waitKey(0)

#1.b)
img = cv2.imread('calvinHobbes.jpeg')
dim=(img.shape[0], img.shape[1])

#Scaling the image up by a factor of 2.5 in both x and y directions
changed_dimension=(int(dim[0] * 2.5), int(dim[1] * 2.5))

#Nearest Neighbor Interpolation
near_img =  cv2.resize(img, changed_dimension, interpolation = cv2.INTER_NEAREST)
cv2.imshow('Nearest Neighbor Interpolation', near_img)
cv2.waitKey(0)

#Bilinear Interpolation
bilinear_img = cv2.resize(img, changed_dimension, interpolation = cv2.INTER_LINEAR)
cv2.imshow('Bilinear Interpolation', bilinear_img)
cv2.waitKey(0)

#Bicubic Interpolation
bicubic_img = cv2.resize(img, changed_dimension, interpolation = cv2.INTER_CUBIC)
cv2.imshow('Bicubic Interpolation', bicubic_img)
cv2.waitKey(0)

#1.c)
trans_image = np.float32([[1, 0, 0.25*dim[0]],[0, 1, 0.5*dim[1]]])
trans_image = cv2.warpAffine(img,trans_image, (dim[0], dim[1]))
cv2.imshow('Transformed image', trans_image)
cv2.waitKey(0)
cv2.destroyAllWindows() 

#2)
#Rotation of image by 45 degree
im_age = cv2.imread('block.png')
cv2.imshow('Original', im_age)
cv2.waitKey(0)
angle=45
new_image=rotate_image(im_age,angle)
cv2.imshow('rotate.png', new_image)
cv2.waitKey(0)

#Scaling the image up by a factor of 10 in both x and y directions
dim=(new_image.shape[0], new_image.shape[1])
changed_dimension=(int(dim[0] * 10), int(dim[1] * 10))

#Nearest Neighbor Interpolation
near_img =  cv2.resize(new_image, changed_dimension, interpolation = cv2.INTER_NEAREST)
cv2.imshow('Nearest Neighbor Interpolation', near_img)
cv2.waitKey(0)
"""
Observation of Nearest Neighbor Interpolation:
This forms a pixelated or blocky image and also, it does not introduce any new data.
"""

#Bilinear Interpolation
bilinear_img = cv2.resize(new_image, changed_dimension, interpolation = cv2.INTER_LINEAR)
cv2.imshow('Bilinear Interpolation', bilinear_img)
cv2.waitKey(0)
"""
Observation of Bilinear Interpolation:
This produces a smooth image than the nearest neighbor but the results for sharp 
transitions like edges are not ideal because the results are a weighted average.
"""

#Bicubic Interpolation
bicubic_img = cv2.resize(new_image, changed_dimension, interpolation = cv2.INTER_CUBIC)
cv2.imshow('Bicubic Interpolation', bicubic_img)
cv2.waitKey(0)
"""
Observation of Bicubic Interpolation:
This produces a sharper image than the above 2 methods and also this 
method balances processing time and output quality fairly well.
"""
cv2.destroyAllWindows() 

#3.a)
image3 = cv2.imread('8.jpg')
cv2.imshow('Original', image3)
cv2.waitKey(0)
angle=45
new_image3=rotate_image(image3,angle)
for i in range(7):
    new_image3=rotate_image(new_image3,angle)
cv2.imshow('New image 3', new_image3)
cv2.waitKey(0)
"""
Here the image dimension changes from 256x256 to 
610x610 and 8 becomes very small
"""

#3.b)
angle=90
new_image4=rotate_image(image3,angle)
for i in range(3):
    new_image4=rotate_image(new_image4,angle)
cv2.imshow('New image 4', new_image4)
cv2.waitKey(0)
cv2.destroyAllWindows() 
"""
Image is same just the angle are different after rotation
"""

#4)
img_new = cv2.imread('chDistorted.jpeg')
rows,cols,ch = img_new.shape

pts1 = np.float32([[4,3],[213,48],[69,199],[234,230]])
pts2 = np.float32([[3,4],[235,3],[3,231],[236,232]])

transformed_matrix = cv2.getPerspectiveTransform(pts1,pts2)

new_img = cv2.warpPerspective(img_new,transformed_matrix,(cols,rows))

plt.subplot(121),plt.imshow(img_new),plt.title('Input')
plt.subplot(122),plt.imshow(new_img),plt.title('Output')
plt.show()

cv2.destroyAllWindows() 

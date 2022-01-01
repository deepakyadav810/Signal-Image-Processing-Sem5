import cv2
import numpy as np
from matplotlib import pyplot as plt
# Q1
# 1)
img = cv2.imread('Fig09_5.tif')
result = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
result = np.array(result, np.uint8)
er = cv2.erode(img, result)
cv2.imwrite('1a.tif', er)
dil = cv2.dilate(img, result)
cv2.imwrite('1b.tif', dil)
morphop = cv2.morphologyEx(img, cv2.MORPH_OPEN, result)
cv2.imwrite('1c.tif', morphop)
morphcl = cv2.morphologyEx(img, cv2.MORPH_CLOSE, result)
cv2.imwrite('1d.tif', morphcl)

# 2)
img = cv2.imread('Fig09_7.tif')
result = np.ones((3, 3), np.uint8)
morphcl = cv2.morphologyEx(img, cv2.MORPH_CLOSE, result)
cv2.imwrite('2.tif', morphcl)

# 3)
img = cv2.imread('Fig09_11.tif')
result = np.ones((5, 5), np.uint8)
morphop = cv2.morphologyEx(img, cv2.MORPH_OPEN, result)
morphop = cv2.morphologyEx(morphop, cv2.MORPH_CLOSE, result)
cv2.imwrite('3.tif', morphop)

# 4)
img = cv2.imread('Fig09_16.tif')
result = np.ones((5, 5), np.uint8)
er = cv2.erode(img, result)
outline = img - er
cv2.imwrite('4.tif', outline)

# Q2

# 1)
bubfish=cv2.imread('bubblingFish.jpg')
grayimg = cv2.cvtColor(bubfish, cv2.COLOR_BGR2GRAY)
edged = cv2.Canny(grayimg, 30, 200)
contours, hierarchy  = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(bubfish, contours, -1, (0, 255, 0), 3)
cv2.imshow('Largest Contour', bubfish)
cv2.imwrite('bubblingFish_contour.jpg', bubfish)

# 2)
img = cv2.imread('bubblingFish.jpg')
Large = max(contours, key = cv2.contourArea)
cv2.drawContours(img, [Large], -1, (0, 255, 0), 3)
cv2.imwrite('bubblingFish_largestContour.jpg', img)

# Q3
polygons=cv2.imread('polygons.png')
grayimg = cv2.cvtColor(polygons, cv2.COLOR_BGR2GRAY)
ret, imagethreshold = cv2.threshold(grayimg, 245, 255, cv2.THRESH_BINARY_INV)
cv2.imwrite("Imagethreshold.png", imagethreshold)
cv2.waitKey(0)
contours, hierarchy = cv2.findContours(imagethreshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
tmp=0
for count in contours:
    epsilon = 0.01499 * cv2.arcLength(count, True)
    approx = cv2.approxPolyDP(count, epsilon, True)
    i, j = approx[0][0]

    if(len(approx) == 3):
        tmp+=1
        cv2.drawContours(polygons, [approx], 0, (0), 3)
print(tmp)
cv2.imwrite("Resulting_polygons.png", polygons)

#Q4
#HoughLines()
sudoku=cv2.imread('sudoku.jpg')
grayimg = cv2.cvtColor(sudoku, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(grayimg,50,150,apertureSize = 3)
lines = cv2.HoughLines(edges,1,np.pi/180,200)
for line in lines:
    for rho,theta in line:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))

        cv2.line(sudoku,(x1,y1),(x2,y2),(0,0,255),2)
cv2.imwrite('houghlines.jpg',sudoku)

#HoughLinesP()
sudoku = cv2.imread('sudoku.jpg')
grayimg = cv2.cvtColor(sudoku, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(grayimg,50,150,apertureSize = 3)
minLineLength = 100
maxLineGap = 10
lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
for line in lines:
    for x1,y1,x2,y2 in line:
        cv2.line(sudoku,(x1,y1),(x2,y2),(0,255,0),2)
cv2.imwrite('houghlinesP.jpg',sudoku)

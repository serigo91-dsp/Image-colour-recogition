import cv2
import os
import numpy
from matplotlib import pyplot


image_1_path = 'IP Images/org_1.png'
img = cv2.imread(image_1_path, 0)

cv2.imshow('Image', img)
img = cv2.GaussianBlur(img, (3,3), cv2.BORDER_DEFAULT)
cv2.waitKey(0)

ret, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

contoured_img = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

cv2.imshow('Image', thresh)
cv2.waitKey(0)
cv2.imshow('Image', thresh2)
cv2.waitKey(0)
cv2.destroyAllWindows()
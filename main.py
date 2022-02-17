import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
import glob


# The HSV threshold will be set here, for later use. This will allow us to change the colour ranges easily

####*******BLUE COLOUR THRESHHOLD**********************
blue_threshold_H_HI = 135
blue_threshold_S_HI = 255
blue_threshold_V_HI = 255
blue_threshold_H_LO = 90
blue_threshold_S_LO = 80
blue_threshold_V_LO = 200

###********GREEN COLOUR THRESHHOLD**********************
green_threshold_H_HI = 70
green_threshold_S_HI = 255
green_threshold_V_HI = 255
green_threshold_H_LO = 45
green_threshold_S_LO = 100
green_threshold_V_LO = 200

#********RED COLOUR THRESHHOLD**********************
red_threshold_H_HI = 20
red_threshold_S_HI = 255
red_threshold_V_HI = 255
red_threshold_H_LO = 0
red_threshold_S_LO = 100
red_threshold_V_LO = 200

#********YELLOW COLOUR THRESHHOLD**********************
yellow_threshold_H_HI = 30
yellow_threshold_S_HI = 255
yellow_threshold_V_HI = 255
yellow_threshold_H_LO = 28
yellow_threshold_S_LO = 100
yellow_threshold_V_LO = 200

#********WHITE COLOUR THRESHHOLD**********************
white_threshold_H_HI = 0
white_threshold_S_HI = 0
white_threshold_V_HI = 255
white_threshold_H_LO = 0
white_threshold_S_LO = 0
white_threshold_V_LO = 200

#*************************************************************************

# Set arrays for storing all images

images = []
images_bw = []

# Find original circle coordinates so we can project future images to fit this frame

img_circles_bw = cv2.imread("IP Images/org_1.png", 0)
img_circles = cv2.imread("IP Images/org_1.png", cv2.IMREAD_COLOR)
img_circles_bw = cv2.GaussianBlur(img_circles_bw, (3, 3), cv2.BORDER_DEFAULT)

circles_coordinates = cv2.HoughCircles(image=img_circles_bw, method=cv2.HOUGH_GRADIENT, dp=1.2, minDist=250, param1=150,
                                       param2=50, minRadius=0, maxRadius=0)

print(circles_coordinates)

for circle in circles_coordinates[0]:
    (x, y, radius) = circle
    radius = int(radius)
    x = int(x)
    y = int(y)
    cv2.circle(img_circles, (x, y), radius, (0, 255, 0), 4)

    cv2.rectangle(img_circles, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

cv2.imshow('Image', img_circles)
cv2.waitKey(1000)

# Load all the images in the folder to be processed

for filename in os.listdir("IP Images"):
    image = cv2.imread(os.path.join("IP Images", filename), cv2.IMREAD_COLOR)
    image_bw = cv2.imread(os.path.join("IP Images", filename), 0)
    if image is not None:
        images.append(image)
    if image_bw is not None:
        images_bw.append(image_bw)

# Start While loop that will run until a "A' Key is pressed

while 1:

    if cv2.waitKey(1000) == ord('a'):
        cv2.destroyAllWindows()
        break
    # Iterate through all the images

    for i in range(len(images)):

        img = images[i]
        img3 = images[i]
        img2 = images_bw[i]

        # Extracting the saturation channel

        sat_channel = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1]

        # For colour recognition we will use the HSV colour space,
        # which has the biggest range of the colour spaces.

        hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        hsv_image = cv2.bilateralFilter(hsv_image, 6, 10, 10, cv2.BORDER_DEFAULT)

        # Now we will binarize the image in order to extract objects for further processing

        img2 = cv2.GaussianBlur(img2, (3, 3), cv2.BORDER_DEFAULT)

        ret, thresh = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY)
        ret2, thresh2 = cv2.threshold(img2, 40, 255, cv2.THRESH_BINARY_INV)

        thresh2 = cv2.GaussianBlur(thresh2, (3, 3), cv2.BORDER_DEFAULT)

        contoured_img, hierarchy = cv2.findContours(thresh2, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        sorted_contours = sorted(contoured_img, key=cv2.contourArea, reverse=True)

        # for pic, c in enumerate(sorted_contours):
        #     area = cv2.contourArea(c)
        #     #   print(area)
        #     if 1900 < area < 10000:
        #         area = cv2.contourArea(c)
        #         x, y, w, h = cv2.boundingRect(c)
        #         imageFrame = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 0), 2)
        #
        # cv2.imshow('Image', img)
        # cv2.waitKey(0)

        # print(sorted_contours)

        plt.subplot(2, 3, 1)
        cv2.imshow('Image', img)
        cv2.waitKey(1000)

        # We will build masks for each of the colours that we can use to compare the colour image with
        # For each colour, we will dilate the pixel so it will only detect the colour
        # and then we will do a bitwise compare.
        kernel = np.ones((8, 8), "uint8")

        # For Blue Colour
        blue_lower = np.array([blue_threshold_H_LO, blue_threshold_S_LO, blue_threshold_V_LO], np.uint8)
        blue_higher = np.array([blue_threshold_H_HI, blue_threshold_S_HI, blue_threshold_V_HI], np.uint8)
        blue_mask = cv2.inRange(hsv_image, blue_lower, blue_higher)
        blue_mask = cv2.dilate(blue_mask, kernel)
        res_blue = cv2.bitwise_and(img, img, mask=blue_mask)
        plt.subplot(2, 3, 2)
        cv2.imshow('Blue Mask', res_blue)
        cv2.waitKey(3000)

        # For Green Colour
        green_lower = np.array([green_threshold_H_LO, green_threshold_S_LO, green_threshold_V_LO], np.uint8)
        green_higher = np.array([green_threshold_H_HI, green_threshold_S_HI, green_threshold_V_HI], np.uint8)
        green_mask = cv2.inRange(hsv_image, green_lower, green_higher)
        green_mask = cv2.dilate(green_mask, kernel)
        res_green = cv2.bitwise_and(img, img, mask=green_mask)
        plt.subplot(2, 3, 3)
        cv2.imshow('Green Mask', res_green)
        cv2.waitKey(3000)

        # For Red Colour
        red_lower = np.array([red_threshold_H_LO, red_threshold_S_LO, red_threshold_V_LO], np.uint8)
        red_higher = np.array([red_threshold_H_HI, red_threshold_S_HI, red_threshold_V_HI], np.uint8)
        red_mask = cv2.inRange(hsv_image, red_lower, red_higher)
        red_mask = cv2.dilate(red_mask, kernel)
        res_red = cv2.bitwise_and(img, img, mask=red_mask)
        plt.subplot(2, 3, 4)
        cv2.imshow('Red Mask', res_red)
        cv2.waitKey(3000)

        # For Yellow Colour
        yellow_lower = np.array([yellow_threshold_H_LO, yellow_threshold_S_LO, yellow_threshold_V_LO], np.uint8)
        yellow_higher = np.array([yellow_threshold_H_HI, yellow_threshold_S_HI, yellow_threshold_V_HI], np.uint8)
        yellow_mask = cv2.inRange(hsv_image, yellow_lower, yellow_higher)
        yellow_mask = cv2.dilate(yellow_mask, kernel)
        res_yellow = cv2.bitwise_and(img, img, mask=yellow_mask)
        plt.subplot(2, 3, 5)
        cv2.imshow('Yellow Mask', res_yellow)
        cv2.waitKey(3000)

        # For White Colour
        white_lower = np.array([white_threshold_H_LO, white_threshold_S_LO, white_threshold_V_LO], np.uint8)
        white_higher = np.array([white_threshold_H_HI, white_threshold_S_HI, white_threshold_V_HI], np.uint8)
        white_mask = cv2.inRange(hsv_image, white_lower, white_higher)
        white_mask = cv2.dilate(white_mask, kernel)
        res_white = cv2.bitwise_and(img, img, mask=white_mask)
        plt.subplot(2, 3, 6)
        cv2.imshow('White Mask', res_white)
        cv2.waitKey(3000)

        blue_img = img.copy()
        green_img = img.copy()
        red_img = img.copy()
        yellow_img = img.copy()
        white_img = img.copy()

        contours_blue, hierarchy_blue = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours_blue):
            area = cv2.contourArea(contour)
            if 1900 < area < 10000:
                x, y, w, h = cv2.boundingRect(contour)
                imageFrame = cv2.rectangle(blue_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(blue_img, "Blue Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX,1.0, (0, 0, 0))

        plt.subplot(2, 3, 1)
        cv2.imshow('Image', blue_img)
        cv2.waitKey(1000)

        contours_green, hierarchy_green = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours_green):
            area = cv2.contourArea(contour)
            if 1900 < area < 10000:
                x, y, w, h = cv2.boundingRect(contour)
                imageFrame = cv2.rectangle(green_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(green_img, "Green Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX,1.0, (0, 0, 0))

        plt.subplot(2, 3, 1)
        cv2.imshow('Image', green_img)
        cv2.waitKey(1000)

        contours_red, hierarchy_red = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours_red):
            area = cv2.contourArea(contour)
            if 100 < area < 10000:
                x, y, w, h = cv2.boundingRect(contour)
                imageFrame = cv2.rectangle(red_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(red_img, "Red Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX,1.0, (0, 0, 0))

        plt.subplot(2, 3, 1)
        cv2.imshow('Image', red_img)
        cv2.waitKey(1000)

        contours_yellow, hierarchy_yellow = cv2.findContours(yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours_yellow):
            area = cv2.contourArea(contour)
            if 1900 < area < 10000:
                x, y, w, h = cv2.boundingRect(contour)
                imageFrame = cv2.rectangle(yellow_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(yellow_img, "Yellow Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX,1.0, (0, 0, 0))

        plt.subplot(2, 3, 1)
        cv2.imshow('Image', yellow_img)
        cv2.waitKey(1000)

        contours_white, hierarchy_white = cv2.findContours(white_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for pic, contour in enumerate(contours_white):
            area = cv2.contourArea(contour)
            if 1900 < area < 10000:
                x, y, w, h = cv2.boundingRect(contour)
                imageFrame = cv2.rectangle(white_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(white_img, "White Colour", (x, y), cv2.FONT_HERSHEY_SIMPLEX,1.0, (0, 0, 0))

        plt.subplot(2, 3, 1)
        cv2.imshow('Image', white_img)
        cv2.waitKey(3000)

        # cv2.imshow('Image', thresh)
        # cv2.waitKey(0)
        # cv2.imshow('Image', thresh2)
        # cv2.waitKey(0)
        # cv2.imshow('Image', img)
        # cv2.waitKey(0)
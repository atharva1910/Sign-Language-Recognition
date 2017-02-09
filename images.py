import cv2
import os
import numpy as np



def takeimage():
    cap = cv2.VideoCapture(0)
    while(cap.isOpened()):
        name = 97
        ret, image = cap.read()
        cv2.rectangle(image, (500, 500), (100, 100), (0, 255, 0), 1)

        # crop image
        img = image[100:500, 100:500]  # x, y, w, h

        # convert to greyscale
        grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # blur the image
        blur_img = cv2.GaussianBlur(grey_img, (35, 35), 0)
        # convert into black and white image
        _, final_img = cv2.threshold(blur_img, 127, 225,
                                     cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        if(cv2.waitKey(10) == 27):
            break
        else:
            filename = str(chr(name) + ".png")
            cv2.imwrite(filename, final_img)
            name = + 1


def imageshow(img):
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # blur the image
    blur_img = cv2.GaussianBlur(image, (35, 35), 0)

    # convert into black and white image
    _, final_img = cv2.threshold(blur_img, 127, 225,
                                 cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    return imageProcessing(final_img, img)


def makeDatabase(*args):
    f = open("dataset", "a")
    line = ''
    for i in args:
        line += str(i) + "::"
    f.write(line)


def split():
    f = open("dataset", "r")
    var1 = []
    var2 = []
    var3 = []
    var4 = []
    var5 = []
    for i in range(26):
        for line in f:
            line = line.split("::")
            var1.append(line[0])
            var2.append(float(line[1]))
            var3.append(int(line[2]))
            var4.append(float(line[3]))
            var5.append(float(line[4]))
    return var1, var2, var3, var4, var5

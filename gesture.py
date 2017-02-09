import cv2
import numpy as np
from images import makeDatabase, split
from scikit-images import hog


def findHOG(image):
    return hog(image)


def matchImage(area, defects_count, max_contour):
    temp = []
    _, radius = cv2.minEnclosingCircle(max_contour)
    length = cv2.arcLength(max_contour, True)
    temp.append(findmin(area, area_arr))
    temp.append(findmin(defects_count, defects_arr))
    temp.append(findmin(length, length_arr))
    temp.append(findmin(radius, radius_arr))
    print(temp)
    return max(set(temp), key=temp.count)


def printDatabase(defects_arr):
    name = 97
    for i in range(len(defects_arr)):
        print(defects_arr[i])
        print(dict[chr(name)])
        name += 1


def findmin(value, arr):
    """
    Returns the closest value to the letter
    """
    min_index = min(arr, key=lambda x: abs(value-x))
    return arr.index(min_index)


def imageProcessing(final_img, img):

    """
    Find the contours of the image and  hulls around the image
    Input : Procssed black and white image
    Output : The max countour found on the image and the image of the
    contour and hull
    """
    defects_count = 0
    image, contour, hierarchy = cv2.findContours(final_img.copy(),
                                                 cv2.RETR_TREE,
                                                 cv2.CHAIN_APPROX_NONE)
    max_contour = max(contour, key=lambda x: cv2.contourArea(x))
    hull = cv2.convexHull(max_contour)

    # draw the contours
    cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 0)

    # find the defects
    hull = cv2.convexHull(max_contour, returnPoints=False)
    return cv2.boundingRect(max_contour)


def init_ui():

    """
    The main program loop, in this loop the screen is recorded and
    image processing funcitons are called
    """
    name = 97
    while(cap.isOpened()):
        ret, image = cap.read()
#        image = fgbg.apply(frame)
        # properties of rec -> img, vertex1, 2, color, thickness
        cv2.rectangle(image, (500, 500), (100, 100), (0, 255, 0), 1)

        # crop image
        img = image[100:500, 100:500]  # x, y, w, h
        keyPressed = cv2.waitKey(10)

        # convert to greyscale
        grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # blur the image
        blur_img = cv2.GaussianBlur(grey_img, (35, 35), 0)
        # convert into black and white image
        _, final_img = cv2.threshold(blur_img, 127, 225,
                                     cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        contour_img, contour_area, defects_count, max_contour \
            = imageProcessing(final_img, img)

        # Display both the contour image and video frame side by side
        display = np.hstack((contour_img, img))
        answer = matchImage(contour_area, defects_count, max_contour)
        print(answer)
        # cv2.putText(img, str(answer+97), (50, 50),
        # cv2.FONT_HERSHEY_SIMPLEX, 2, 255)

        if(keyPressed == 27):
            cleanup()

        elif(keyPressed == 32):
            filename = chr(name)
            makeDatabase
            print("wrting " + filename)
            name += 1

        cv2.imshow("Hand Recogntition", display)


def cleanup():

    """
    All the cleanup regarding the program is done in this function
    such as closing the video imput feed etc.
    """

    cap.release()
    cv2.destroyAllWindows()
    return


if __name__ == "__main__":
    dict = {}
    char_name, area_arr, defects_arr, length_arr, radius_arr = split()
#    printDatabase()
    cap = cv2.VideoCapture(0)
    fgbg = cv2.createBackgroundSubtractorMOG2()
    print("Press esc to exit")
    init_ui()

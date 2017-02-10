import cv2
import numpy as np
from images import makeDatabase, split
import math


def recognize(contour_area, defects_count, centroid, left, right, top):

    """
    Recognize the gesture based on diffrent parameters
    """
    if defects_count == 2:
        angle = findAngle(left, top, centroid)
        print(angle)
        return "2"
    return defects_count + 1


def findAngle(start, end, far):

    """
    Find the angle between the points
    """
    a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
    b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
    c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
    angle = math.acos((b**2 + c**2 - a**2)/(2*b*c)) * 57
    return angle


def applyBackgroundSubtractor(image):

    """
    Remove the background from the image
    """
    frame = fgbg.apply(image)
    kernel = np.ones((3, 3), np.uint8)
    fg_mask = cv2.erode(frame, kernel, iterations=1)
    image = cv2.bitwise_and(frame, frame, mask=fg_mask)
    return image


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
    cv2.drawContours(img, [hull], 0, (255, 255, 255), 0)
    cv2.drawContours(img, [max_contour], 0, (255, 255, 255), 0)

    # Fine the contour area, extreme left and right points
    contour_area = cv2.contourArea(max_contour)
    extLeft = tuple(max_contour[max_contour[:, :, 0].argmin()][0])
    extRight = tuple(max_contour[max_contour[:, :, 0].argmax()][0])
    extTop = tuple(max_contour[max_contour[:, :, 1].argmin()][0])

    # find the defects
    hull = cv2.convexHull(max_contour, returnPoints=False)
    defects = cv2.convexityDefects(max_contour, hull)
    if defects is not None:
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i][0]
            start = tuple(max_contour[s][0])
            end = tuple(max_contour[e][0])
            far = tuple(max_contour[f][0])
            angle = findAngle(start, end, far)
            if angle <= 90:
                defects_count += 1
                cv2.circle(img, far, 1, [255, 0, 255], 1)
                cv2.circle(img, start, 1, [255, 0, 255], 1)
                cv2.circle(img, end, 1, [255, 0, 255], 1)

            moments = cv2.moments(max_contour)
            # Central mass of first order moments
            if moments['m00'] != 0:
                cx = int(moments['m10']/moments['m00'])  # cx = M10/M00
                cy = int(moments['m01']/moments['m00'])  # cy = M01/M00
                centerMass = (cx, cy)
                cv2.circle(img, centerMass, 7, [100, 0, 255], 2)

    cv2.imshow("image", img)

    answer = recognize(contour_area, defects_count, centerMass,
                       extLeft, extRight, extTop)
    return answer


def init_ui():

    """
    The main program loop, in this loop the screen is recorded and
    image processing funcitons are called
    """
    c_pressed = False
    while(cap.isOpened()):
        ret, image = cap.read()

        # properties of rec -> img, vertex1, 2, color, thickness
        cv2.rectangle(image, (550, 550), (100, 100), (0, 255, 0), 1)

        # crop image
        crop_img = image[100:550, 100:550]  # x, y, w, h
        keyPressed = cv2.waitKey(10)

        # convert to greyscale
        grey_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)

        # blur the image
        blur_img = cv2.GaussianBlur(grey_img, (35, 35), 0)

        # convert into black and white image
        _, final_img = cv2.threshold(blur_img, 127, 225,
                                     cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        thresh_img = cv2.erode(final_img, None, iterations=2)
        final_img = cv2.dilate(thresh_img, None, iterations=2)

        if c_pressed:
            answer = imageProcessing(final_img, crop_img)
        else:
            answer = "Press C to capture"

        cv2.putText(image, str(answer), (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, 255)

        cv2.imshow("Hand detection", image)

        # Keyboard handling
        if(keyPressed == 27):
            cleanup()

        elif(keyPressed == ord('c')):
            c_pressed = True


def cleanup():

    """
    All the cleanup regarding the program is done in this function
    such as closing the video imput feed etc.
    """

    cap.release()
    cv2.destroyAllWindows()
    return


if __name__ == "__main__":
    # dict = {}
    # char_name, area_arr, defects_arr, length_arr, radius_arr = split()
    # printDatabase()
    cap = cv2.VideoCapture(0)
    fgbg = cv2.createBackgroundSubtractorMOG2()
    print("Press esc to exit")
    init_ui()

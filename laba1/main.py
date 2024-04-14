import cv2
import imutils
import numpy as np

#демонстрація зображення
def show_i(name, img):
    cv2.imshow(name, img)
    cv2.waitKey()
    cv2.destroyAllWindows()


img = cv2.imread('image.jpg')




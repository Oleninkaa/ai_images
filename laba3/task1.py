import cv2

from main import process_image

low_t = 50
hight_t = 150
y_bottom = 160
y_upper = 400
vertices = [(305, 166),(367, 166)]

info = low_t,hight_t,y_bottom,y_upper,vertices
img = cv2.imread('road2.jpg')

process_image(img, info)
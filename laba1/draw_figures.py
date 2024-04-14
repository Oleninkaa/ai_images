from main import show_i, img
import cv2
import imutils
import numpy as np


cv2.rectangle(img, (150,170), (450, 70), (0,0,255), 3)
show_i("shinji", img)



cv2.line(img, (0,0), (610,468), (255,0,0), 3)
show_i("shinji", img)



points = np.array([[198, 231], [315, 311], [400, 233], [198, 231]])
cv2.polylines(img, np.int32([points]), 1, (255, 255, 255), 3)
show_i("shinji", img)



cv2.circle(img, (397, 289), 85, (0, 0, 255), 3)
show_i("shinji", img)



font = cv2.FONT_HERSHEY_SCRIPT_COMPLEX
cv2.putText(
    img, 'evangelion', (30,420), font, 2, (255,255,255), 4, cv2.LINE_4
)
show_i("shinji", img)
# Завантаження відео
import cv2

from main import process_image

video_capture = cv2.VideoCapture('road2.mp4')

low_t = 50
hight_t = 150
y_bottom = 310
y_upper = 500
vertices = [(305, 166),(367, 166)]

info = low_t,hight_t,y_bottom,y_upper,vertices

while video_capture.isOpened():
    ret, frame = video_capture.read()
    if ret:
        process_image(frame, info, 2)
        cv2.imshow('frame', frame)

        if cv2.waitKey(1) & 0xff == ord('q'):
            break
    else:
        break

video_capture.release()
cv2.destroyAllWindows()
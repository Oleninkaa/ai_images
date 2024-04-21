import cv2
import numpy

face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
cv2.startWindowThread()

cap = cv2.VideoCapture('people_walking.mp4')

while True:
    ret, frame = cap.read()

    frame = cv2.resize(frame, (800,560))
    gray_filter = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    boxes, weights = hog.detectMultiScale(frame, winStride=(8,8))
    boxes = numpy.array([[x,y, x+w, y+h] for (x,y,w,h) in boxes])

    for (x, y, w, h) in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 1)
        roi_human = frame[y:y + h, x:x + w]
        gray_roi = cv2.cvtColor(roi_human, cv2.COLOR_BGR2GRAY)
        face_rects = face_cascade.detectMultiScale(gray_roi, scaleFactor=1.1, minNeighbors=5)

        for (xf, yf, wf, hf) in face_rects:
            cv2.rectangle(frame, (x + xf, y + yf), (x + xf + wf, y + yf + hf), (255, 0, 0), 2)

    cv2.imshow('Video', frame)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break

cap.release()
cv2.destroyAllWindows()
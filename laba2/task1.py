import cv2

face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
eye_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')


scaling_factor = 0.5
frame = cv2.imread('img/skz1.jpg')

gray_filter = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


# frame = cv2.resize(frame, None, fx = scaling_factor, fy = scaling_factor, interpolation=cv2.INTER_AREA)
face_rects=face_cascade.detectMultiScale(gray_filter, scaleFactor=1.1, minNeighbors=5)

print(f"Found {len(face_rects)} faces!")
for (x, y,w,h) in face_rects:
    cv2.rectangle(frame,(x,y),(x+w, y+h), (255,0,0), 2)
    roi_gray = gray_filter[y:y+h, x:x+w]
    roi_color = frame[y:y + h, x:x + w]
    smile=smile_cascade.detectMultiScale(roi_gray)
    eye=eye_cascade.detectMultiScale(roi_gray)

    for(sx,sy,sw,sh) in smile:
        cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 1)

    for (ex, ey, ew, eh) in eye:
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 1)

cv2.imshow('Example', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
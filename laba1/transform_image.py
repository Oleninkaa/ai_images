from main import show_i, img
import cv2
import imutils
import numpy as np


#вивести кольорову комбінацію пікселя
(blue, green, red) = img[100,50]
print(f"{red = }, {green = }, {blue = }")

#вирізання фрагменту зображення
piece = img[60:260, 320:689]

#збереження зображення
cv2.imwrite('saved/img.jpg', img)

#зміна розміру зображення за допомогою ratio
h,w = img.shape[0:2]
h_new = 321
ratio = w/h
w_new=int(h_new*ratio)
resized_1 = cv2.resize(img,(w_new,h_new))

#зміна розміру зображення за допомогою imutils
resized_2 = imutils.resize(img, width = 321)

#поворот зображення
h,w = resized_1.shape[0:2]
center = (w//2, h//2)
M = cv2.getRotationMatrix2D(center, -45, 1)
rotated_1 = cv2.warpAffine(resized_1, M, (w,h))

#поворот зображення imutils
rotated_2 = imutils.rotate(resized_2, 45)

#розмите зображення
blurred = cv2.GaussianBlur(resized_2, (11,11), 0)

#склеювання
collage = np.hstack((resized_2, blurred))


show_i("shinji", img)
show_i("piece shinji", piece)
show_i("resized shinji 1", resized_1)
show_i("resized shinji 2", resized_2)

show_i("rotated shinji 1", rotated_1)
show_i("rotated shinji 2", rotated_2)

show_i("blurred shinji", blurred)

show_i("collage shinji", collage)
from utils import face_encodings
import cv2
image = cv2.imread(
    "Junichiro_Koizumi/Junichiro_Koizumi_0002.jpg"
)
print(face_encodings(image))
print(face_encodings(image)[0].shape)

import os
import glob
import pickle
import cv2
from matplotlib import pyplot as plt

from utils import face_encodings, nb_of_matches, face_rects

# load the encodings + names dictionary
with open("encodings.pickle", "rb") as f:
    name_encodings_dict = pickle.load(f)

# get the list of image files in the examples directory
image_files = glob.glob("examples/*.jpg")

for image_file in image_files:
    # load the input image
    image = cv2.imread(image_file)

    # get the 128-d face embeddings for each face in the input image
    encodings = face_encodings(image)
    # this list will contain the names of each face detected in the image
    names = []

    # loop over the encodings
    for encoding in encodings:
        # initialize a dictionary to store the name of the
        # person and the number of times it was matched
        counts = {}
        # loop over the known encodings
        for (name, encodings) in name_encodings_dict.items():
            # compute the number of matches between the current
            # encoding and the encodings
            # of the known faces and store the number of matches
            # in the dictionary
            counts[name] = nb_of_matches(encodings, encoding)
        # check if all the number of matches are equal to 0
        # if there is no match for any name, then we set the name
        # to "Unknown"
        if all(count == 0 for count in counts.values()):
            name = "Unknown"
        # otherwise, we get the name with the highest number of matches
        else:
            name = max(counts, key=counts.get)

        # add the name to the list of names
        names.append(name)

    # loop over the `rectangles` of the faces in the
    # input image using the `face_rects` function
    for rect, name in zip(face_rects(image), names):
        # get the bounding box for each face using the `rect` variable
        x1, y1, x2, y2 = rect.left(), rect.top(), rect.right(), rect.bottom()
        # draw the bounding box of the face along with the name of the person
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, name, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    # add the filename as a title at the top of the image
    filename = os.path.basename(image_file)
    cv2.putText(image, filename, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # show the output image
    cv2.imshow("image", image)
    cv2.waitKey(0)

    # save the output image
    output_image_path = f"output_{filename}"
    cv2.imwrite(output_image_path, image)

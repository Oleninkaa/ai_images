import cv2
import  numpy as np
import  imutils

def read_file(filename):
    img = cv2.imread(filename)
    img = imutils.resize(img, width=500)
    cv2.imshow("hyunjin.jpg", img)
    cv2.imwrite('img.jpg', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return img

def edge_mask(img, line_size, blur_value):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_blur = cv2.medianBlur(gray, blur_value)
    edges = cv2.adaptiveThreshold(gray_blur,
                                  255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY,
                                  line_size,
                                  blur_value)
    cv2.imshow("Img", edges)
    cv2.imwrite('edge_mask.jpg', edges)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return edges
line_size = 7
blur_value = 5

edges = edge_mask(read_file('hyunjin.jpg'), line_size, blur_value)


def color_quantization(img, k):
    # Transform the image
    data = np.float32(img).reshape((-1, 3))

    # Determine criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)

    # Implementing K-Means
    ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(img.shape)

    cv2.imshow("Color quantization", result)
    cv2.imwrite('color_quantization.jpg', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return result
total_color = 3

cq = color_quantization(read_file('hyunjin.jpg'), total_color)


def make_blurred(img):
    blurred = cv2.bilateralFilter(img, d=5, sigmaColor=300, sigmaSpace=200)
    cv2.imshow("Blurred", blurred)
    cv2.imwrite('make_blurred.jpg', blurred)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return blurred


blurred = make_blurred(cq)


def make_cartoon(blurred_, edges_):
    cartoon_ = cv2.bitwise_and(blurred_, blurred_, mask=edges_)
    cv2.imshow("Cartoon", cartoon_)
    cv2.imwrite('make_cartoon.jpg', cartoon_)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return cartoon_

make_cartoon(blurred, edges)

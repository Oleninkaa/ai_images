import cv2
import  numpy as np
import  imutils

def read_file(filename):
    img = cv2.imread(filename)
    img = imutils.resize(img, width=500)
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
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return edges





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
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return result





def make_blurred(img):
    blurred = cv2.bilateralFilter(img, d=5, sigmaColor=300, sigmaSpace=200)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return blurred





def make_cartoon(blurred_, edges_):
    cartoon_ = cv2.bitwise_and(blurred_, blurred_, mask=edges_)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return cartoon_



def cartoon_image(image):
    line_size = 7
    blur_value = 5
    total_color = 3
    edges = edge_mask(image, line_size, blur_value)
    cq = color_quantization(image, total_color)
    blurred = make_blurred(cq)
    make_cartoon(blurred, edges)
    return make_cartoon(blurred, edges)




cap = cv2.VideoCapture('video.mp4')

if not cap.isOpened():
    print("Could not open video file")



frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(frame_width)
print(frame_height)
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter('output_video1.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    cartoon_img = cartoon_image(frame)
    out.write(cartoon_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

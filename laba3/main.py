import cv2
import numpy as np

def process_image(frame, info, mode=1):
    low_t, hight_t, y_bottom, y_upper, vertices = info
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    kernel_size = 5
    blur = cv2.GaussianBlur(grayscale, (kernel_size, kernel_size), 0)

    # low_t = 50
    # hight_t = 150
    edges = cv2.Canny(blur, low_t, hight_t)

    vertices = np.array([[(0, frame.shape[0]), vertices[0],vertices[1], (frame.shape[1], frame.shape[0])]], dtype=np.int32)
    mask = np.zeros_like(edges)
    ignore_mask_color = (255, 255, 255)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_edges = cv2.bitwise_and(edges, mask)

    rho = 3
    theta = np.pi / 180
    threshold = 15
    min_line_len = 150
    max_line_gap = 60
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)

    draw_lines(frame, lines, y_bottom, y_upper, mode)

def draw_lines(img, lines, y_bottom, y_upper, mode, color=[255, 0, 0], thickness=7,):
    x_bottom_pos = []
    x_upper_pos = []
    x_bottom_neg = []
    x_upper_neg = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            if 0.5 < slope < 0.8:
                b = y1 - slope * x1
                x_bottom_pos.append((y_bottom - b) / slope)
                x_upper_pos.append((y_upper - b) / slope)
            elif -0.8 < slope < -0.5:
                b = y1 - slope * x1
                x_bottom_neg.append((y_bottom - b) / slope)
                x_upper_neg.append((y_upper - b) / slope)

    lines_mean = []
    if x_bottom_pos and x_upper_pos:
        lines_mean.append([int(np.mean(x_bottom_pos)), int(np.mean(y_bottom)), int(np.mean(x_upper_pos)), int(np.mean(y_upper))])
    if x_bottom_neg and x_upper_neg:
        lines_mean.append([int(np.mean(x_bottom_neg)), int(np.mean(y_bottom)), int(np.mean(x_upper_neg)), int(np.mean(y_upper))])

    for line_mean in lines_mean:
        cv2.line(img, (line_mean[0], line_mean[1]), (line_mean[2], line_mean[3]), color, thickness)
    if (mode==1):
        cv2.imshow('input', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()





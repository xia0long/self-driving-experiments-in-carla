import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def gray(img):
    
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def gaussian_blur(img, kernel_size=3):
    
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def canny(img):
    low_threshold = 60
    high_threshold = 150
    canny_img = cv2.Canny(img, low_threshold, high_threshold)
    
    return canny_img

def region_of_interest(img):
    scale_w = 7 / 16
    scale_h = 11 /18
    height, width = img.shape
    left_bottom = [0, height - 1]
    right_bottom = [width - 1, height - 1]
    left_up = [scale_w * width, scale_h * height]
    right_up = [(1 - scale_w) * width, scale_h * height]
    vertices = np.array([[left_bottom, left_up, right_up, right_bottom]], dtype=np.int32)

    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, [vertices], ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)

    return masked_image

def pre_processing(img):
    img = gray(img)
    img = gaussian_blur(img)
    img = canny(img)
    img = region_of_interest(img)

    return img

if __name__ == "__main__":
    img = cv2.imread("demo.jpg")
    img_processed = pre_processing(img)
    plt.imshow(img_processed)
    plt.show()

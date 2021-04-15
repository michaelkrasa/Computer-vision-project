import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def convolve_3x3(image, kernel):
    (imgHeight, imgWidth) = image.shape[:2]

    # set pad size and copy image with adding the padding
    pad_size = 1
    image = cv.copyMakeBorder(image, pad_size, pad_size, pad_size, pad_size, cv.BORDER_REPLICATE)
    output = np.zeros((imgHeight, imgWidth), dtype="float32")  # use floats for best precision

    for y in np.arange(pad_size, imgHeight + pad_size):
        for x in np.arange(pad_size, imgWidth + pad_size):
            # get the region of the pixels we're interested in
            region = image[y - pad_size:y + pad_size + 1, x - pad_size:x + pad_size + 1]

            # doing the kernel multiplication of the region and summing it
            k = (region * kernel).sum()
            # store the convolved values in the output image
            output[y - pad_size, x - pad_size] = k

    return output


# Open JPEG image
img = cv.imread('kitty.bmp', cv.COLOR_BGR2GRAY)

# Initialising kernels
kernel_ones = np.ones((3, 3), np.float32) / 9

kernel_weighted = np.ones((3, 3), np.float32)
kernel_weighted[1][1] = 10  # weighted kernel center
kernel_weighted /= 18

# Task 1
convolved_img = convolve_3x3(img, kernel_ones)
weighted_convolved_img = convolve_3x3(img, kernel_weighted)

cv.imwrite('kitty3x3.bmp', convolved_img)  # 2D convolution
cv.imwrite('kitty3x3-weighted.bmp', weighted_convolved_img)

# Task 2
sobel_x = cv.Sobel(convolved_img, cv.CV_64F, 1, 0, ksize=3)
sobel_y = cv.Sobel(convolved_img, cv.CV_64F, 0, 1, ksize=3)
cv.imwrite('histogram_x.bmp', sobel_x)
cv.imwrite('histogram_y.bmp', sobel_y)

edge_strength_img = cv.add(sobel_x, sobel_y)  # combining images
cv.imwrite('histogram_combined.bmp', edge_strength_img)  # edge strength image

# Task 3
plt.hist(edge_strength_img.ravel(),256,[0,256]); plt.show()  # plotting histogram
xy, threshold_img = cv.threshold(edge_strength_img, 90, 255, cv.THRESH_BINARY)
cv.imwrite('threshold.bmp', threshold_img)  # threshold image

# Task 4
sobel_x_w = cv.Sobel(weighted_convolved_img, cv.CV_64F, 1, 0, ksize=3)
sobel_y_w = cv.Sobel(weighted_convolved_img, cv.CV_64F, 0, 1, ksize=3)
edge_strength_img_w = cv.add(sobel_x_w, sobel_y_w)  # combining images

xy, weighted_threshold_img = cv.threshold(edge_strength_img_w, 90, 255, cv.THRESH_BINARY)
cv.imwrite('threshold_weighted.bmp', weighted_threshold_img)  # weighted threshold image


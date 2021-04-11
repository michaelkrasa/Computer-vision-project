import cv2 as cv
import numpy as np

# Open JPEG image
img = cv.imread('kitty.bmp', cv.COLOR_BGR2GRAY)


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


# Initialising kernels
kernel_ones = np.ones((3, 3), np.float32) / 9

kernel_weighted = np.ones((3, 3), np.float32)
kernel_weighted[1][1] = 10  # weighted kernel center
kernel_weighted /= 18

# Save image in PNG format
cv.imwrite('kitty3x3.bmp', convolve_3x3(img, kernel_ones))  # 2D convolution
cv.imwrite('kitty3x3-weighted.bmp', convolve_3x3(img, kernel_weighted))

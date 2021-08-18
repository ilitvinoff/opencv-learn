import queue

import cv2 as cv

# Load two images
import opencv_scripts

img1 = cv.imread(opencv_scripts.test_images['legion'])
img2 = cv.imread(opencv_scripts.test_images['bob'])
# I want to put logo on top-left corner, So I create a ROI
rows, cols, channels = img2.shape
roi = img1[0:rows, 0:cols]
# Now create a mask of logo and create its inverse mask also
img2gray = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
ret, mask = cv.threshold(img2gray, 10, 127, cv.THRESH_BINARY)
mask_inv = cv.bitwise_not(mask)
# Now black-out the area of logo in ROI
img1_bg = cv.bitwise_and(roi, roi, mask=mask_inv)
# Take only region of logo from logo image.
img2_fg = cv.bitwise_and(img2, img2, mask=mask)
# Put logo in ROI and modify the main image
dst = cv.add(img1_bg, img2_fg)
img1[0:rows, 0:cols] = dst

img_queue = queue.Queue()
img_queue.put((img1,'res'))

while not img_queue.empty():
    opencv_scripts.handle_img(img_queue.get(),img_queue)

cv.destroyAllWindows()

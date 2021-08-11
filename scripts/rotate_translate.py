import queue

import cv2

# Reading the image
import numpy as np

from scripts import test_images, handle_img

image = cv2.imread(test_images['bob'])

# dividing height and width by 2 to get the center of the image
height, width = image.shape[:2]
# get the center coordinates of the image to create the 2D rotation matrix
center = (width/2, height/2)

# using cv2.getRotationMatrix2D() to get the rotation matrix
rotate_matrix = cv2.getRotationMatrix2D(center=center, angle=-45, scale=1)

# rotate the image using cv2.warpAffine
rotated_image = cv2.warpAffine(src=image, M=rotate_matrix, dsize=(width, height))

# get tx and ty values for translation
# you can specify any value of your choice
tx, ty = width / 4, height / 4

# create the translation matrix using tx and ty, it is a NumPy array
translation_matrix = np.array([
    [1, 0, tx],
    [0, 1, ty]
], dtype=np.float32)

translated_image = cv2.warpAffine(src=image, M=translation_matrix, dsize=(width, height))

img_queue = queue.Queue()
img_queue.put((image, "origin"))
img_queue.put((rotated_image, "rotated"))
img_queue.put((translated_image, "translated"))

while not img_queue.empty():
    img = img_queue.get()
    handle_img(img, img_queue)

cv2.destroyAllWindows()
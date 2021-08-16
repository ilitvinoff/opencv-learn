import queue
import sys

import cv2

from opencv_scripts import test_images, handle_img

img_tuple = ((cv2.imread(test_images['legion'], cv2.IMREAD_COLOR), 1),
             (cv2.imread(test_images['legion'], cv2.IMREAD_GRAYSCALE), 2),
             (cv2.imread(test_images['legion'], cv2.IMREAD_UNCHANGED), 3))

img_queue = queue.Queue()

for img_config in img_tuple:
    img_queue.put(img_config)

while not img_queue.empty():
    img_config = img_queue.get()
    handle_img(img_config, img_queue)

cv2.destroyAllWindows()

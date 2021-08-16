# Import packages
import queue

import cv2

from opencv_scripts import test_images, handle_img

img = cv2.imread(test_images['two cubes'])
print(f'orig size: {img.shape}')  # Print image shape

cropped_image = img[0:img.shape[0], int(img.shape[1]/2):]
print(f'cropped size: {cropped_image.shape}')  # Print image shape

img_queue = queue.Queue()
img_queue.put((img, "origin"))
img_queue.put((cropped_image, "cropped"))

while not img_queue.empty():
    img = img_queue.get()
    handle_img(img, img_queue,window_flag=None)

cv2.destroyAllWindows()

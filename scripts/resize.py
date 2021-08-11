# let's start with the Imports
import queue
import random

import cv2

# Read the image using imread function
from scripts import test_images, handle_img

image_legion = cv2.imread(test_images['legion'])
cv2.imshow('Legion', image_legion)

image_bob = cv2.imread(test_images['bob'])
cv2.namedWindow('Sponge Bob', cv2.WINDOW_NORMAL)
cv2.imshow('Sponge Bob', image_bob)

# let's downscale the image using new  width and height
down_width = 300
down_height = 200
down_points = (down_width, down_height)
resized_down = (
cv2.resize(image_legion, down_points, interpolation=cv2.INTER_AREA), 'Resized Down')

# let's upscale the image using new  width and height
up_width = image_bob.shape[1] * 2
up_height = image_bob.shape[0] * 2
up_points = (up_width, up_height)
resized_up = (
cv2.resize(image_bob, up_points, interpolation=cv2.INTER_LINEAR), 'Resized Up')

# Display images
img_queue = queue.Queue()
img_queue.put(resized_up)
img_queue.put(resized_down)

print(f"queue size before cycle {img_queue.qsize()}")

while not img_queue.empty():
    img = img_queue.get()
    handle_img(img,img_queue)

# press any key to close the windows
cv2.destroyAllWindows()

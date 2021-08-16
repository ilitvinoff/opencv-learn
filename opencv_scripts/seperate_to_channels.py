import queue

import numpy as np
import cv2

import opencv_scripts

IMAGE = "/home/i_litvinov/Pictures/openCV-images/opencv_split_merge_merdged_01.png"
image = cv2.imread(IMAGE)
hsv_image= cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
channels = cv2.split(hsv_image)
channels[0].max()
# show each channel individually

img_queue = queue.Queue()
for i, channel in enumerate(channels):
    img_queue.put((channel, f"channel{i}"))

merged = cv2.merge(channels)
img_queue.put((merged,"merged"))

while not img_queue.empty():
    img = img_queue.get()
    opencv_scripts.handle_img(img, img_queue, window_flag=None)
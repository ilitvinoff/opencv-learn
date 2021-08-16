import queue

import cv2
import numpy as np

import opencv_scripts


def average_integer(lst):
    return int(sum(lst) / len(lst))


THRESHOLD = 40

BRIGHT = cv2.imread(opencv_scripts.test_images['light cube'])
DARK = cv2.imread(opencv_scripts.test_images['dark cube'])

BRIGHT_HSV = cv2.cvtColor(BRIGHT, cv2.COLOR_BGR2HSV)
DARK_HSV = cv2.cvtColor(DARK, cv2.COLOR_BGR2HSV)

min, max, bincount = opencv_scripts.get_min_max_bincount(opencv_scripts.test_images['light green'], "HSV")
bright_H, bright_S, bright_V = opencv_scripts.separate_channels(np.array([[min, max, bincount]], dtype='int64'))

print(f"BRIGHT:\nmin: {min}\nmax: {max}\nbincount: {bincount}")
print(f"HSV: [{bright_H}, {bright_S}, {bright_V},]")

min, max, bincount = opencv_scripts.get_min_max_bincount(opencv_scripts.test_images['dark green'], "HSV")
dark_H, dark_S, dark_V = opencv_scripts.separate_channels(np.array([[min, max, bincount]], dtype='int64'))

print(f"DARK:\nmin: {min}\nmax: {max}\nbincount: {bincount}")
print(f"HSV: [{dark_H}, {dark_S}, {dark_V},]")

averageH = average_integer([average_integer(bright_H), average_integer(dark_H)])
averageS = average_integer([average_integer(bright_S), average_integer(bright_S)])
averageV = average_integer([average_integer(bright_V), average_integer(bright_V)])

print(f"average HSV: [{averageH}, {averageS}, {averageV}]")

minHSV = np.array([averageH - THRESHOLD, averageS - THRESHOLD, averageV - THRESHOLD])
maxHSV = np.array([averageH + THRESHOLD, averageS + THRESHOLD, averageV + THRESHOLD])

maskHSV = cv2.inRange(BRIGHT_HSV, minHSV, maxHSV)
resultHSV_bright = cv2.bitwise_and(BRIGHT_HSV, BRIGHT_HSV, mask=maskHSV)

maskHSV = cv2.inRange(DARK_HSV, minHSV, maxHSV)
resultHSV_dark = cv2.bitwise_and(DARK_HSV, DARK_HSV, mask=maskHSV)

channels = cv2.split(BRIGHT)
equ_bright = cv2.merge((cv2.equalizeHist(channels[0]), cv2.equalizeHist(channels[1]),cv2.equalizeHist(channels[2])))

channels = cv2.split(DARK)
equ_dark = cv2.merge((cv2.equalizeHist(channels[0]), cv2.equalizeHist(channels[1]),cv2.equalizeHist(channels[2])))

bright_concatenated = opencv_scripts.concatenate_img([
    [BRIGHT, "original bright"],
    [equ_bright, "equalized"],
    [BRIGHT_HSV, "bright hsv"],
    [resultHSV_bright, "bright"]
])

dark_concatenated = opencv_scripts.concatenate_img([
    [DARK, "original dark"],
    [equ_dark, "equalized"],
    [DARK_HSV, "dark hsv"],
    [resultHSV_dark, "dark"]
])

img_queue = queue.Queue()

img_queue.put((bright_concatenated,"indoor"))
img_queue.put((dark_concatenated,"outdoor"))

while not img_queue.empty():
    img = img_queue.get()
    opencv_scripts.handle_img(img, img_queue, window_flag=None)

cv2.destroyAllWindows()

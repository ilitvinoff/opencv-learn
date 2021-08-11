# Import packages
import queue

import cv2
import numpy as np

from scripts import test_images, handle_img

img_dict = dict()
bright = cv2.imread(test_images['light cube'])
dark = cv2.imread(test_images['dark cube'])

brightLAB = cv2.cvtColor(bright, cv2.COLOR_BGR2LAB)
darkLAB = cv2.cvtColor(dark, cv2.COLOR_BGR2LAB)

brightYCB = cv2.cvtColor(bright, cv2.COLOR_BGR2YCrCb)
darkYCB = cv2.cvtColor(dark, cv2.COLOR_BGR2YCrCb)

brightHSV = cv2.cvtColor(bright, cv2.COLOR_BGR2HSV)
darkHSV = cv2.cvtColor(dark, cv2.COLOR_BGR2HSV)

bgr = [40, 158, 16]
thresh = 40

minBGR = np.array([bgr[0] - thresh, bgr[1] - thresh, bgr[2] - thresh])
maxBGR = np.array([bgr[0] + thresh, bgr[1] + thresh, bgr[2] + thresh])

maskBGR = cv2.inRange(bright, minBGR, maxBGR)
img_dict['resultBGR bright'] = cv2.bitwise_and(bright, bright, mask=maskBGR), 'resultBGR bright'

maskBGR = cv2.inRange(dark, minBGR, maxBGR)
img_dict['resultBGR dark'] = cv2.bitwise_and(dark, dark, mask=maskBGR), 'resultBGR dark'

# convert 1D array to 3D, then convert it to HSV and take the first element
# this will be same as shown in the above figure [65, 229, 158]
hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0][0]

minHSV = np.array([hsv[0] - thresh, hsv[1] - thresh, hsv[2] - thresh])
maxHSV = np.array([hsv[0] + thresh, hsv[1] + thresh, hsv[2] + thresh])

maskHSV = cv2.inRange(brightHSV, minHSV, maxHSV)
img_dict['resultHSV bright'] = cv2.bitwise_and(brightHSV, brightHSV, mask=maskHSV), 'resultHSV bright'

maskHSV = cv2.inRange(darkHSV, minHSV, maxHSV)
img_dict['resultHSV dark'] = cv2.bitwise_and(darkHSV, darkHSV, mask=maskHSV), 'resultHSV dark'

# convert 1D array to 3D, then convert it to YCrCb and take the first element
ycb = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2YCrCb)[0][0]

minYCB = np.array([ycb[0] - thresh, ycb[1] - thresh, ycb[2] - thresh])
maxYCB = np.array([ycb[0] + thresh, ycb[1] + thresh, ycb[2] + thresh])

maskYCB = cv2.inRange(brightYCB, minYCB, maxYCB)
img_dict['resultYCB bright'] = cv2.bitwise_and(brightYCB, brightYCB, mask=maskYCB), 'resultYCB bright'

maskYCB = cv2.inRange(darkYCB, minYCB, maxYCB)
img_dict['resultYCB dark'] = cv2.bitwise_and(darkYCB, darkYCB, mask=maskYCB), 'resultYCB dark'

# convert 1D array to 3D, then convert it to LAB and take the first element
lab = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2LAB)[0][0]

minLAB = np.array([lab[0] - thresh, lab[1] - thresh, lab[2] - thresh])
maxLAB = np.array([lab[0] + thresh, lab[1] + thresh, lab[2] + thresh])

maskLAB = cv2.inRange(brightLAB, minLAB, maxLAB)
img_dict['resultLAB bright'] = cv2.bitwise_and(brightLAB, brightLAB, mask=maskLAB), 'resultLAB bright'

maskLAB = cv2.inRange(darkLAB, minLAB, maxLAB)
img_dict['resultLAB dark'] = cv2.bitwise_and(darkLAB, darkLAB, mask=maskLAB), 'resultLAB dark'

img_dict['bright'] = bright, "orig bright"
img_dict['dark'] = dark, "orig dark"

img_queue = queue.Queue()
for k, v in img_dict.items():
    img_queue.put(v)

while not img_queue.empty():
    img = img_queue.get()
    handle_img(img, img_queue, window_flag=None)

cv2.destroyAllWindows()

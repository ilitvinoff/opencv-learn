import os
import queue

import cv2
from skimage.exposure import is_low_contrast

from opencv_scripts import handle_img

IMAGE_DIR_PATH = "assets/cube"
THRESH = 0.8


def img_list_generator(dir_path):
    for item in os.listdir(dir_path):
        yield item


img_list = img_list_generator(IMAGE_DIR_PATH)

for image in img_list:
    imagePath = os.path.join(IMAGE_DIR_PATH, image)

    # load the input image from disk, resize it, and convert it to
    # grayscale
    image = cv2.imread(imagePath)
    image = cv2.resize(image, dsize=(400, 300))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blur the image slightly and perform edge detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 30, 150)
    # initialize the text and color to indicate that the input image
    # is *not* low contrast
    text = "Low contrast: No"
    color = (0, 0, 255)

    img_copy = image.copy()

    if is_low_contrast(gray, fraction_threshold=THRESH):
        # update the text and color
        text = "Low contrast: Yes"
        # otherwise, the image is *not* low contrast, so we can continue
        # processing it
    else:
        # find contours in the edge map and find the largest one,
        # which we'll assume is the outline of our color correction
        # card
        contours, hierarchy = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)

        cv2.drawContours(image=img_copy, contours=contours, contourIdx=-1, color=(255, 0, 255), thickness=6)

    cv2.putText(img_copy, text, (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                color, 2)

    img_queue = queue.Queue()
    img_queue.put((img_copy, "bordered"))
    while not img_queue.empty():
        handle_img(img_queue.get(), img_queue)

        if img_queue.empty():
            cv2.destroyAllWindows()

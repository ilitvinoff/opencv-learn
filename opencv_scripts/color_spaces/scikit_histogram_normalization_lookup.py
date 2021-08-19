import os
import queue
import sys

import cv2
import numpy as np
from skimage import exposure

from opencv_scripts import test_images, concatenate_img

EXPECTED_SIZE = (309, 300)
THRESHOLD = 40
CHANNEL_BOUNDARIES = (255, 255, 255)

CUBES = "assets/cube/"
PIECES = "assets/pieces/"

def handle_img(img_config, img_queue, window_flag=cv2.WINDOW_NORMAL):
    if img_config[0] is None:
        sys.exit("Could not read the image.")
    cv2.namedWindow(f'{img_config[1]}', window_flag)

    if cv2.getWindowProperty(f'{img_config[1]}', cv2.WND_PROP_VISIBLE) >= 1:
        buffer_img = img_config[0].copy()
        cv2.imshow(f'{img_config[1]}', buffer_img)

        img_queue.put(img_config)
    else:
        cv2.destroyAllWindows()

def convert_size(img, size):
    return cv2.resize(img, size)


def get_most_common_color(img):
    img_temp = img.copy()
    unique, counts = np.unique(img_temp.reshape(-1, 3), axis=0, return_counts=True)
    return unique[np.argmax(counts)].reshape(1, 1, 3)


def get_color_diapason(start_color, threshold, boundaries):
    l = l if (l := start_color[0][0][0] - threshold) > 0 else 0
    a = a if (a := start_color[0][0][1] - threshold) > 0 else 0
    b = b if (b := start_color[0][0][2] - threshold) > 0 else 0
    min = np.array([l, a, b])
    l = l if (l := start_color[0][0][0] + threshold) < boundaries[0] else boundaries[0]
    a = a if (a := start_color[0][0][1] + threshold) < boundaries[1] else boundaries[1]
    b = b if (b := start_color[0][0][2] + threshold) < boundaries[2] else boundaries[2]
    max = np.array([l, a, b])

    return min, max


def detect_color_area(common_color, src, ref):
    mask_model = exposure.match_histograms(src, ref, multichannel=True)
    min, max = get_color_diapason(common_color, THRESHOLD, CHANNEL_BOUNDARIES)
    mask = cv2.inRange(mask_model, min, max)

    return cv2.bitwise_and(src, src, mask=mask)


def clear_queue(obj):
    while not obj.empty():
        obj.get()


def fil_with_color(shape, dtype, color):
    color_img = np.empty(shape=shape, dtype=dtype)
    color_img = color_img.reshape(-1, 3)
    color_img[:] = color
    return color_img.reshape(shape)


def get_panel(src_img, ref_img, ref_piece):
    normalized = exposure.match_histograms(src_img, ref_img, multichannel=True)

    common_color = get_most_common_color(ref_piece)
    masked = detect_color_area(common_color=common_color, src=src_img, ref=ref_img)
    color_img = fil_with_color(masked.shape, masked.dtype, common_color)

    img_config_list = [[ref_img, 'ref img'], [src_img, 'src'], [normalized, 'normalized'], [color_img, 'color'],
                       [masked, 'masked']]

    return concatenate_img(img_config_list)


def key_event_handler(k):
    return k == ord('n'), k == ord('s')


dark_piece = cv2.imread(test_images['dark green'])
light_piece = cv2.imread(test_images['light green'])
light_cube = cv2.imread(test_images["light cube"])
dark_cube = cv2.imread(test_images["dark cube"])

stop_program = False
next_piece = False

for cube in os.listdir(CUBES):
    ref = dark_cube
    src = convert_size(img=cv2.imread(os.path.join(CUBES, cube)),size=ref.shape[:-1][::-1])

    img_queue = queue.Queue()
    img_queue.put((get_panel(src_img=src, ref_img=ref, ref_piece=dark_piece), 'hist normalization'))

    if stop_program:
        clear_queue(img_queue)
        cv2.destroyAllWindows()
        break

    while not img_queue.empty():

        handle_img(img_queue.get(), img_queue)
        k = cv2.waitKey(1)
        next_piece, stop_program = key_event_handler(k)

        if next_piece or stop_program:
            clear_queue(img_queue)
            cv2.destroyAllWindows()
            break

cv2.destroyAllWindows()

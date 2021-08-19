import os
import queue
import sys

import cv2
import numpy as np

import opencv_scripts

HSV_CHANNEL_BOUNDARIES = (179, 255, 255)
THRESHOLD = 5
EXPECTED_SIZE = (309, 300)

CUBES = "assets/cube/"
PIECES = "assets/pieces/"


def average_integer(lst):
    return int(sum(lst) / len(lst))


def get_most_common_color(img):
    img_temp = img.copy()
    unique, counts = np.unique(img_temp.reshape(-1, 3), axis=0, return_counts=True)
    return unique[np.argmax(counts)].reshape(1, 1, 3)


def get_color_diapason(start_color, threshold, boundaries):
    h = h if (h := start_color[0][0][0] - threshold) > 0 else 0
    s = s if (s := start_color[0][0][1] - threshold) > 0 else 0
    v = v if (v := start_color[0][0][2] - threshold) > 0 else 0
    min = np.array([h, s, v])
    h = h if (h := start_color[0][0][0] + threshold) < boundaries[0] else boundaries[0]
    s = s if (s := start_color[0][0][1] + threshold) < boundaries[1] else boundaries[1]
    v = v if (v := start_color[0][0][2] + threshold) < boundaries[2] else boundaries[2]
    max = np.array([h, s, v])

    return min, max


def get_hsv_from_bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


def equalize(img):
    channels = cv2.split(img)
    return cv2.merge((cv2.equalizeHist(channels[0]), cv2.equalizeHist(channels[1]), cv2.equalizeHist(channels[2])))


def convert_size(img):
    return cv2.resize(img, EXPECTED_SIZE)


def detect_color_area(color_hsv, image):
    mask_model = convert_size(cv2.cvtColor(image, cv2.COLOR_BGR2HSV))
    mask_model[:, :, 2] = color_hsv[0][0][2]
    mask_model[:, :, 1] = color_hsv[0][0][1]

    min, max = get_color_diapason(color_hsv, THRESHOLD, HSV_CHANNEL_BOUNDARIES)
    mask = cv2.inRange(mask_model, min, max)
    return cv2.bitwise_and(image, image, mask=mask)


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


def key_event_handler(k):
    return k == ord('k'), k == ord('n'), k == ord('s')


def clear_queue(obj):
    while not obj.empty():
        obj.get()


img_queue = queue.Queue()

# To stop all cycles if need
stop_program = False

# for every variant of color to search for
for piece in os.listdir(PIECES):
    if stop_program:
        clear_queue(img_queue)
        cv2.destroyAllWindows()
        break

    img_piece = cv2.imread(os.path.join(PIECES, piece))

    common_color = (get_most_common_color(get_hsv_from_bgr(img_piece)))

    # move to next piece variant
    next_piece = False

    # for different lightning conditions
    for cube in os.listdir(CUBES):
        if stop_program or next_piece:
            clear_queue(img_queue)
            cv2.destroyAllWindows()
            break

        # show which color we searching now
        cv2.imshow(f"{os.path.split(piece)[1]}", img_piece)

        img_cube = convert_size(cv2.imread(os.path.join(CUBES, cube)))
        result_cube = detect_color_area(common_color, img_cube)

        img_panel = opencv_scripts.concatenate_img([
            [img_cube, "origin cube"],
            [result_cube, "masked cube"]
        ])

        # load queue to start handle img
        img_queue.put((img_panel, f"search for {os.path.split(piece)[1]}"))

        stop_current_show = False
        # show result of the searching
        while not img_queue.empty() and not stop_current_show:

            handle_img(img_queue.get(), img_queue)
            k = cv2.waitKey(1)
            stop_current_show, next_piece, stop_program = key_event_handler(k)

            if next_piece or stop_program or stop_current_show:
                clear_queue(img_queue)
                cv2.destroyAllWindows()
                break


#
# BRIGHT = convert_size(cv2.imread(opencv_scripts.test_images['light cube']))
# DARK = convert_size(cv2.imread(opencv_scripts.test_images['dark cube']))
# BRIGHT_HSV = cv2.cvtColor(BRIGHT, cv2.COLOR_BGR2HSV)
# DARK_HSV = cv2.cvtColor(DARK, cv2.COLOR_BGR2HSV)
#
# common_color_light_HSV = (get_most_common_color(get_hsv_from_bgr(opencv_scripts.test_images['dark green'])))
#
# result_bright = detect_color_area(common_color_light_HSV, BRIGHT)
# result_dark = detect_color_area(common_color_light_HSV, DARK)
#
# bright_concatenated = opencv_scripts.concatenate_img([
#     [BRIGHT, "original bright"],
#     [BRIGHT_HSV, "bright hsv"],
#     [result_bright, "bright"]
# ])
#
# dark_concatenated = opencv_scripts.concatenate_img([
#     [DARK, "original dark"],
#     [DARK_HSV, "dark hsv"],
#     [result_dark, "dark"]
# ])
#
# img_queue = queue.Queue()
#
# img_queue.put((bright_concatenated, "indoor"))
# img_queue.put((dark_concatenated, "outdoor"))
#
# while not img_queue.empty():
#     img = img_queue.get()
#     opencv_scripts.handle_img(img, img_queue, window_flag=None)

import sys
from functools import reduce

import cv2
import numpy as np

test_images = {'legion': "/home/i_litvinov/Pictures/10.jpg",
               'bob': "/home/i_litvinov/Pictures/07.jpg",
               'two cubes': "/home/i_litvinov/Pictures/openCV-images/two-cubes.png",
               'dark cube': "/home/i_litvinov/Pictures/openCV-images/dark_cube.jpg",
               'light cube': "/home/i_litvinov/Pictures/openCV-images/light_cube.jpg",
               'dark green': "/home/i_litvinov/Pictures/openCV-images/dark_green.jpg",
               'light green': "/home/i_litvinov/Pictures/openCV-images/light_green.jpg",
               }


def show_normal_window(window_title, img):
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.imshow(window_title, img)


def handle_img(img_config, img_queue, window_flag=cv2.WINDOW_NORMAL, isRectangle=False):
    if img_config[0] is None:
        sys.exit("Could not read the image.")
    cv2.namedWindow(f'{img_config[1]}', window_flag)

    if cv2.getWindowProperty(f'{img_config[1]}', cv2.WND_PROP_VISIBLE) >= 1:
        buffer_img = img_config[0].copy()
        start_point = (0, 0)
        thickness = 10
        end_point = (buffer_img.shape[1], buffer_img.shape[0])

        if isRectangle:
            cv2.rectangle(buffer_img, start_point, end_point, (0, 255, 255), thickness=thickness, lineType=cv2.LINE_8)

        cv2.imshow(f'{img_config[1]}', buffer_img)
        k = cv2.waitKey(1)

        if k == ord("s"):
            cv2.imwrite(f"legion{img_config[1]}.jpg", img_config[0])

        img_queue.put(img_config)


def add_border_and_title_to_img(img_config, thickness, border_color, title_color):
    result = cv2.copyMakeBorder(img_config[0], thickness, thickness, thickness, thickness, cv2.BORDER_CONSTANT,
                                value=border_color)

    header = np.zeros((100, result.shape[1], 3), np.uint8)
    header[:] = border_color
    header = cv2.copyMakeBorder(header, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    result = cv2.copyMakeBorder(result, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    result = np.vstack((header, result))

    cv2.putText(img=result, text=img_config[1], org=(40, 40), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                color=title_color, thickness=3, lineType=0)

    return result


def concatenate_img(img_config_list, border_thickness=10, border_color=(255, 255, 255), title_color=(128, 128, 128)):
    buffer = img_config_list.copy()

    buffer = list(map(lambda a: add_border_and_title_to_img(a, border_thickness, border_color, title_color), buffer))
    result = np.array(reduce(lambda a, b: np.concatenate((a, b), axis=1), buffer))

    return result


def separate_channels(ndarray):
    c1 = np.array([], dtype='int64')
    c2 = np.array([], dtype='int64')
    c3 = np.array([], dtype='int64')

    b = ndarray[:, :, 0]
    b = b.reshape(b.shape[0] * b.shape[1])
    g = ndarray[:, :, 1]
    g = g.reshape(g.shape[0] * g.shape[1])
    r = ndarray[:, :, 2]
    r = r.reshape(r.shape[0] * r.shape[1])
    c1 = np.append(c1, b)
    c2 = np.append(c2, g)
    c3 = np.append(c3, r)

    return c1, c2, c3


def get_separate_BGR_channels(img):
    return separate_channels(cv2.imread(img))


def get_separate_HSV_channels(img):
    return separate_channels(cv2.imread(img))


separate_channels_methods = {
    "BGR": get_separate_BGR_channels,
    "HSV": get_separate_HSV_channels
}


def get_min_max_bincount(img, method):
    c1, c2, c3 = separate_channels_methods[method](img)

    return [min(c1), min(c2), min(c3)], [max(c1), max(c2), max(c3)], [
        np.bincount(c1).argmax(), np.bincount(c2).argmax(), np.bincount(c3).argmax()]


def str_min_max_bincount(img, method):
    min, max, bincount = get_min_max_bincount(img, method)

    return (f"min: [{min}];\nmax: [{max}];\nbincount: [{bincount}]")

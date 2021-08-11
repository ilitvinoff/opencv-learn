import sys

import cv2

test_images = {'legion': "/home/i_litvinov/Pictures/10.jpg",
               'bob': "/home/i_litvinov/Pictures/07.jpg",
               'two cubes':"/home/i_litvinov/Pictures/openCV-images/two-cubes.png",
               'dark cube':"/home/i_litvinov/Pictures/openCV-images/dark_cube.jpg",
               'light cube':"/home/i_litvinov/Pictures/openCV-images/light_cube.jpg"
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

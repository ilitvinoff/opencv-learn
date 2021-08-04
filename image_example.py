import concurrent.futures
import sys
import threading

import cv2
import queue


def handle_img(img_config, img_queue):
    pass
    if img_config[0] is None:
        sys.exit("Could not read the image.")
    cv2.namedWindow(f'{img_config[1]}', cv2.WINDOW_NORMAL)

    if cv2.getWindowProperty(f'{img_config[1]}', cv2.WND_PROP_VISIBLE) >= 1:
        start_point = (0, 0)
        end_point = (img_config[0].shape[0]-1, img_config[0].shape[1]-1)

        cv2.rectangle(img_config[0], start_point, end_point, (0, 0, 255), thickness=10, lineType=cv2.LINE_8)

        k = cv2.waitKey(1)

        if k == ord("s"):
            cv2.imwrite(f"legion{img_config[1]}.jpg", img_config[0])

        img_queue.put(img_config[0])


img_tuple = ((cv2.imread("/home/i_litvinov/Pictures/10.jpg", cv2.IMREAD_COLOR),1),
             (cv2.imread("/home/i_litvinov/Pictures/10.jpg", cv2.IMREAD_GRAYSCALE),2),
             (cv2.imread("/home/i_litvinov/Pictures/10.jpg", cv2.IMREAD_UNCHANGED),3))

img_queue = queue.Queue()

for img_config in img_tuple:
    img_queue.put(img_config)

while not img_queue.empty():
    img_config = img_queue.get()
    handle_img(img_config,img_queue)

cv2.destroyAllWindows()

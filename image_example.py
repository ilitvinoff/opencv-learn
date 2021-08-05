import queue
import sys

import cv2


def handle_img(img_config, img_queue):
    if img_config[0] is None:
        sys.exit("Could not read the image.")
    cv2.namedWindow(f'{img_config[1]}', cv2.WINDOW_NORMAL)

    if cv2.getWindowProperty(f'{img_config[1]}', cv2.WND_PROP_VISIBLE) >= 1:
        buffer_img = img_config[0].copy()
        start_point = (0, 0)
        thickness = 10
        end_point = (buffer_img.shape[1], buffer_img.shape[0])

        cv2.rectangle(buffer_img, start_point, end_point, (0, 255, 255), thickness=thickness, lineType=cv2.LINE_8)

        cv2.imshow(f'{img_config[1]}', buffer_img)
        k = cv2.waitKey(1)

        if k == ord("s"):
            cv2.imwrite(f"legion{img_config[1]}.jpg", img_config[0])

        img_queue.put(img_config)


img_tuple = ((cv2.imread("/home/i_litvinov/Pictures/10.jpg", cv2.IMREAD_COLOR), 1),
             (cv2.imread("/home/i_litvinov/Pictures/10.jpg", cv2.IMREAD_GRAYSCALE), 2),
             (cv2.imread("/home/i_litvinov/Pictures/10.jpg", cv2.IMREAD_UNCHANGED), 3))

img_queue = queue.Queue()

for img_config in img_tuple:
    img_queue.put(img_config)

while not img_queue.empty():
    img_config = img_queue.get()
    handle_img(img_config, img_queue)

cv2.destroyAllWindows()

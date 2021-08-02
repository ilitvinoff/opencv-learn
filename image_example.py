import concurrent.futures
import sys
import threading

import cv2


def handle_img(name, img):
    if img is None:
        sys.exit("Could not read the image.")
    cv2.namedWindow(f'{name}', cv2.WINDOW_NORMAL)
    cv2.imshow(f'{name}', img)

    while cv2.getWindowProperty(f'{name}', cv2.WND_PROP_VISIBLE) >= 1:
        k = cv2.waitKey(1)

        if k == ord("s"):
            cv2.imwrite(f"legion{name}.jpg", img)
            break


img_tuple = (cv2.imread("/home/i_litvinov/Pictures/legion.jpg", cv2.IMREAD_COLOR),
             cv2.imread("/home/i_litvinov/Pictures/legion.jpg", cv2.IMREAD_GRAYSCALE),
             cv2.imread("/home/i_litvinov/Pictures/legion.jpg", cv2.IMREAD_UNCHANGED))


with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    i = 1
    futures = []
    for img in img_tuple:
        futures.append(executor.submit(handle_img, i, img))
        i += 1

    for future in futures:
        future.result()

cv2.destroyAllWindows()

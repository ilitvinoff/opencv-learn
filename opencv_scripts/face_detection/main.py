import os
import queue

import cv2

# Load the cascade
from opencv_scripts import handle_img, concatenate_img

FACE_CASCADE = cv2.CascadeClassifier('assets/haarcascade_frontalface_default.xml')
INPUT_IMAGES_DIR = "assets/test_images"


def img_list_generator(dir_path):
    for item in os.listdir(dir_path):
        yield os.path.join(dir_path, item)

def detect_faces(img_path):
    # Read the input image
    img = cv2.imread(img_path)
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = FACE_CASCADE.detectMultiScale(gray, 1.1, 4)
    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.rectangle(gray, (x, y), (x + w, y + h), (255, 0, 0), 2)

    gray = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
    img = concatenate_img([(gray,'gray'),(img, 'colored')])

    return img


input_img_path_generator = img_list_generator(INPUT_IMAGES_DIR)

for img_path in input_img_path_generator:
    face_marked_image = detect_faces(img_path)

    img_queue = queue.Queue()
    img_queue.put((face_marked_image, "face detection"))

    while not img_queue.empty():
        handle_img(img_queue.get(), img_queue)

        if img_queue.empty():
            cv2.destroyAllWindows()

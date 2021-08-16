
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import PIL
import opencv_scripts

IMG_1 = opencv_scripts.test_images['light green']
IMG_2 = opencv_scripts.test_images['dark green']


def show_img_compare(img_1, img_2 ):
    f, ax = plt.subplots(1, 2, figsize=(10,10))
    ax[0].imshow(img_1)
    ax[1].imshow(img_2)
    ax[0].axis('off') #hide the axis
    ax[1].axis('off')
    f.tight_layout()
    plt.show()

img = cv.imread(IMG_1)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
img_2 = cv.imread(IMG_2)
img_2 = cv.cvtColor(img_2, cv.COLOR_BGR2RGB)

dim = (600, 600)
# resize image
img = cv.resize(img, dim, interpolation = cv.INTER_AREA)
img_2 = cv.resize(img_2, dim, interpolation = cv.INTER_AREA)

show_img_compare(img, img_2)

img_temp = img.copy()
unique, counts = np.unique(img_temp.reshape(-1, 3), axis=0, return_counts=True)
img_temp[:,:,0], img_temp[:,:,1], img_temp[:,:,2] = unique[np.argmax(counts)]

img_temp_2 = img_2.copy()
unique, counts = np.unique(img_temp_2.reshape(-1, 3), axis=0, return_counts=True)
img_temp_2[:,:,0], img_temp_2[:,:,1], img_temp_2[:,:,2] = unique[np.argmax(counts)]

show_img_compare(img, img_temp)
show_img_compare(img_2, img_temp_2)
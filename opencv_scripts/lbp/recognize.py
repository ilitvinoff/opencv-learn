# import the necessary packages
import os
import queue

import cv2
import numpy as np
from imutils import paths
from skimage import feature
from sklearn.svm import LinearSVC

# construct the argument parse and parse the arguments
from opencv_scripts import handle_img

TRAINING_IMAGES = "/home/i_litvinov/py-projects/opneCV_study/opencv_scripts/lbp/images/training/"
TEST_IMAGES = "/home/i_litvinov/py-projects/opneCV_study/opencv_scripts/lbp/images/testing/"


class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius

    def describe(self, image, eps=1e-7):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        lbp = feature.local_binary_pattern(image, self.numPoints,
                                           self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
                                 bins=np.arange(0, self.numPoints + 3),
                                 range=(0, self.numPoints + 2))
        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)
        # return the histogram of Local Binary Patterns
        return hist


desc = LocalBinaryPatterns(24, 8)
data = []
labels = []

# loop over the training images
for imagePath in paths.list_images(TRAINING_IMAGES):
    # load the image, convert it to grayscale, and describe it
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = desc.describe(gray)
    # extract the label from the image path, then update the
    # label and data lists
    labels.append(imagePath.split(os.path.sep)[-2])
    data.append(hist)
# train a Linear SVM on the data
model = LinearSVC(C=500.0, random_state=13)
model.fit(data, labels)

for imagePath in paths.list_images(TEST_IMAGES):
    # load the image, convert it to grayscale, describe it,
    # and classify it
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = desc.describe(gray)
    prediction = model.predict(hist.reshape(1, -1))

    # display the image and the prediction
    cv2.putText(image, prediction[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (0, 0, 255), 3)

    img_queue = queue.Queue()
    img_queue.put((image, os.path.basename(imagePath)))

    while not img_queue.empty():
        handle_img(img_queue.get(), img_queue)

        if img_queue.empty():
            cv2.destroyAllWindows()

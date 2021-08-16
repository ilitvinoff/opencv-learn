import cv2
from matplotlib import pyplot as plt

IMAGE = "/home/i_litvinov/Pictures/openCV-images/opencv_split_merge_merdged_01.png"

image = cv2.imread(IMAGE)
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# img_queue = queue.Queue()
# img_queue.put((image, "orig"))

hist = cv2.calcHist(images=[image], channels=[0], mask=None, histSize=[256], ranges=[0, 256])

# while not img_queue.empty():
#     img = img_queue.get()
#     opencv_scripts.handle_img(img, img_queue, window_flag=None)


# create a new figure and then plot a 2D color histogram for the
# green and blue channels
fig = plt.figure()
ax = fig.add_subplot(131)
hist = cv2.calcHist([chans[1], chans[0]], [0, 1], None, [32, 32],
	[0, 256, 0, 256])
p = ax.imshow(hist, interpolation="nearest")
ax.set_title("2D Color Histogram for G and B")
plt.colorbar(p)
# plot a 2D color histogram for the green and red channels
ax = fig.add_subplot(132)
hist = cv2.calcHist([chans[1], chans[2]], [0, 1], None, [32, 32],
	[0, 256, 0, 256])
p = ax.imshow(hist, interpolation="nearest")
ax.set_title("2D Color Histogram for G and R")
plt.colorbar(p)
# plot a 2D color histogram for blue and red channels
ax = fig.add_subplot(133)
hist = cv2.calcHist([chans[0], chans[2]], [0, 1], None, [32, 32],
	[0, 256, 0, 256])
p = ax.imshow(hist, interpolation="nearest")
ax.set_title("2D Color Histogram for B and R")
plt.colorbar(p)
# finally, let's examine the dimensionality of one of the 2D
# histograms
print("2D histogram shape: {}, with {} values".format(
	hist.shape, hist.flatten().shape[0]))

hist = cv2.calcHist([image], [0, 1, 2],
	None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
print("3D histogram shape: {}, with {} values".format(
	hist.shape, hist.flatten().shape[0]))

# display the original input image
plt.figure()
plt.axis("off")
plt.imshow(imutils.opencv2matplotlib(image))
# show our plots
plt.show()
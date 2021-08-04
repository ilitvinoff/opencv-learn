# Create a video capture object, in this case we are reading the video from a file
import cv2

vid_capture = cv2.VideoCapture('/home/i_litvinov/Downloads/Telegram Desktop/IMG_0323.MOV')

if (vid_capture.isOpened() == False):
    print("Error opening the video file")
# Read fps and frame count
else:
    # Get frame rate information
    # You can replace 5 with CAP_PROP_FPS as well, they are enumerations
    fps = vid_capture.get(5)
    print('Frames per second : ', fps, 'FPS')

    # Get frame count
    # You can replace 7 with CAP_PROP_FRAME_COUNT as well, they are enumerations
    frame_count = vid_capture.get(7)
    print('Frame count : ', frame_count)

while (vid_capture.isOpened()):

    # vid_capture.read() methods returns a tuple, first element is a bool
    # and the second is frame
    ret, frame = vid_capture.read()

    # 20 is in milliseconds, try to increase the value, say 50 and observe
    key = cv2.waitKey(500)

    if key == ord('q') or cv2.getWindowProperty('Frame', cv2.WND_PROP_VISIBLE) < 1:
        break

    if ret:
        cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)
        cv2.imshow('Frame', frame)


# Release the video capture object
vid_capture.release()
cv2.destroyAllWindows()

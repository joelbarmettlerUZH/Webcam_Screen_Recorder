import numpy as np
import cv2
from PIL import ImageGrab, ImageColor
import screeninfo
import time

def distort_perspective(framegrabber):
    tries = 30*10
    showImage(grid=False)
    while tries >= 0:
        tries -= 1
        gray = framegrabber(slice_median(getWebcamFrame()), getWebcamFrame(), treshold=0.5)
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.bilateralFilter(gray, 11, 17, 17)
        edged = cv2.Canny(gray, 30, 200)
        modified_image, contours, hieararchy = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        # loop over our contours
        for contour in contours:
            # approximate the contour
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            # if our approximated contour has four points, then
            # we can assume that we have found our screen
            if len(approx) == 4:
                screenCnt = approx
                tries = -1
                break
    cv2.destroyAllWindows()
    ###Perspective distortion
    points = screenCnt.reshape(4, 2)
    rect = [None] * 4
    s = points.sum(axis=1)

    tl_index = np.argmin(s)
    rect[0] = (points[tl_index])

    br_index = np.argmax(s)
    rect[2] = points[br_index]
    points = np.delete(points, max(br_index, tl_index), 0)
    points = np.delete(points, min(br_index, tl_index), 0)

    tr_index = np.argmin(list(zip(points[0], points[1]))[0])
    rect[3] = (points[tr_index])
    points = np.delete(points, tr_index, 0)

    rect[1] = points[0]

    (tl, tr, br, bl) = rect
    rect = np.array(rect, np.float32)

    #Work with real information
    downscale_factor = 2
    screen_id = global_screen
    screen = screeninfo.get_monitors()[screen_id]
    width, height = screen.width//downscale_factor, screen.height//downscale_factor

    # construct our destination points which will be used to
    # map the screen to a top-down, "birds eye" view
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]], dtype="float32")

    # calculate the perspective transform matrix and warp
    # the perspective to grab the screen
    M = cv2.getPerspectiveTransform(rect, dst)
    return M, (width, height)


def getScreenFrame():
    img = ImageGrab.grab(bbox=None)
    img = np.array(img)
    return img

def getMaskedScreen(colour, frame, treshold=0.5):
    print(colour)
    lowerBound = colour*(1-treshold)
    upperBound = colour*(1+treshold)
    mask = cv2.inRange(frame, lowerBound, upperBound)
    kernelOpen = np.ones((5, 5))
    kernelClose = np.ones((20, 20))
    maskOpen = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernelOpen)
    maskClose = cv2.morphologyEx(maskOpen, cv2.MORPH_CLOSE, kernelClose)
    return maskClose

def getWebcamFrame():
    ret, frame = cap.read()
    return frame

def showImage(grid=False):
    screen_id = global_screen
    # get the size of the screen
    screen = screeninfo.get_monitors()[screen_id]
    width, height = screen.width, screen.height

    # create image
    greenscreen = np.ones((height, width, 3), dtype=np.float32)
    greenscreen[:, :] = [0, 1, 0]
    if grid:
        greenscreen[::50, :] = [1,0,1]
        greenscreen[:, ::50] = [1,0,1]

    cv2.namedWindow('projector', cv2.WND_PROP_FULLSCREEN)
    cv2.moveWindow('projector', screen.x - 1, screen.y - 1)
    cv2.setWindowProperty('projector', cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    cv2.imshow('projector', greenscreen)
    cv2.waitKey(1)

def slice_median(frame, size=20):
    frame_height = len(frame)
    frame_width = len(frame[0])
    frame_slice = frame[frame_height // 2 - 20:frame_height // 2 + 20, frame_width // 2 - 20:frame_width // 2 + 20]
    frame_slice = frame_slice.reshape([len(frame_slice) * len(frame_slice[0]), 3])
    median_colour = np.median(frame_slice, axis=0)
    return median_colour

def initialize_camera():
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    showImage(grid=True)
    frame = getWebcamFrame()
    frame_height = len(frame)
    frame_width = len(frame[0])
    while frame_width < 1024:
        frame_width *= 1.2
        frame_height *= 1.2
    cap.set(3, int(frame_width))
    cap.set(4, int(frame_height))
    frame = getWebcamFrame()
    frame_height = len(frame)
    frame_width = len(frame[0])
    while (True):
        frame = getWebcamFrame()
        cv2.putText(frame, "Cover field with hand:", (frame_width // 2 - 130, frame_height // 2 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)
        cv2.rectangle(frame, (frame_width // 2 - 20, frame_height // 2 - 20),
                      (frame_width // 2 + 20, frame_height // 2 + 20), (0, 0, 255), 1)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    global hand_colour
    hand_colour = slice_median(frame)
    while (True):
        frame = getWebcamFrame()
        cv2.putText(frame, "Remove hand again:", (frame_width // 2 - 130, frame_height // 2 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    #cv2.destroyAllWindows()
    time.sleep(3)
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

def findFinger():
    global hand_colour
    while(True):
        webcam_frame = cv2.warpPerspective(getWebcamFrame(), M, (maxWidth, maxHeight))
        mask = getMaskedScreen(hand_colour, webcam_frame, treshold=0.5)
        cv2.imshow("normal", webcam_frame)
        cv2.imshow("mask", mask)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    pass

hand_colour = None
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
global_screen = 1
initialize_camera()
M, (maxWidth, maxHeight) = distort_perspective(getMaskedScreen)
findFinger()
while(False):
    # Capture frame-by-frame

    frame = getWebcamFrame()
    cv2.imshow('frame', frame)

    #display = cv2.warpPerspective(frame, M, (maxWidth, maxHeight))
    #cv2.imshow('webcam',display) #place frame here

    #screen = getScreenFrame()
    #cv2.imshow('screen',screen) #place frame here

    #mask = getMaskedScreen()
    #cv2.imshow('mask', mask)  # place frame here

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
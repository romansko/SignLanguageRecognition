#!/usr/bin/env python3
"""
Dataset building utility.
Based on https://github.com/jgv7/CNN-HowManyFingers/blob/master/application.py
Script captures photos to directories. [A->Z].
Mapping Hebrew AlphaBet to English AlphaBet by letter order.

Controls:
use arrows to move the ROI box.
press 1 to turn capturing on/off.
press 2 to display/hide binary mask.
press english letters to choose destination folder.
press esc to exit.

@author: Netanel Azoulay
@author: Roman Koifman
"""

import os
import copy
import cv2
from utils import binaryMask
from projectParams import classes

# Globals
dataColor = (0, 255, 0)
font = cv2.FONT_HERSHEY_SIMPLEX
className = classes[0]
count = 0

dataFolder = 'captureData'  # The data folder to save the captured images to.


def initClass(name):
    global className, count
    className = name
    try:
        os.makedirs(dataFolder + '/%s' % name)
    except OSError as e:
        print(e)
    finally:
        count = len(os.listdir(dataFolder + '/%s' % name))


def captureImages():
    """
    Main loop.
    """

    global font
    global takingData, dataColor
    global className, count
    global showMask

    showMask = 0
    takingData = 0
    fx, fy, fh = 10, 50, 45
    x0, y0, width = 200, 220, 224

    cam = cv2.VideoCapture(0)
    cv2.namedWindow('Original', cv2.WINDOW_NORMAL)

    while True:
        # Get camera frame
        ret, frame = cam.read()
        frame = cv2.flip(frame, 1)  # mirror
        window = copy.deepcopy(frame)
        cv2.rectangle(window, (x0, y0), (x0 + width - 1, y0 + width - 1), dataColor, 12)

        # draw text
        if takingData:
            dataColor = (0, 250, 0)
            cv2.putText(window, 'Data Taking: ON', (fx, fy), font, 1.2, dataColor, 2, 1)
        else:
            dataColor = (0, 0, 250)
            cv2.putText(window, 'Data Taking: OFF', (fx, fy), font, 1.2, dataColor, 2, 1)
        cv2.putText(window, 'Class Name: %s (%d)' % (className, count), (fx, fy + fh), font, 1.0, (245, 210, 65), 2, 1)

        # get region of interest
        roi = frame[y0:y0 + width, x0:x0 + width]
        roi = binaryMask(roi)

        # apply processed roi in frame
        if showMask:
            window[y0:y0 + width, x0:x0 + width] = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)

        # take data or apply predictions on ROI
        if takingData:
            cv2.imwrite(dataFolder + '/{0}/{0}_{1}.png'.format(className, count), roi)
            count += 1

        # show the window
        cv2.imshow('Original', window)

        # Keyboard inputs
        key = cv2.waitKeyEx(10)

        # use ESC key to close the program
        if key & 0xff == 27:
            break

        elif key & 0xff == 255:  # nothing pressed
            continue

        # adjust the position of window
        elif key == 2490368:  # up
            y0 = max((y0 - 5, 0))
        elif key == 2621440:  # down
            y0 = min((y0 + 5, window.shape[0] - width))
        elif key == 2424832:  # left
            x0 = max((x0 - 5, 0))
        elif key == 2555904:  # right
            x0 = min((x0 + 5, window.shape[1] - width))

        key = key & 0xff

        # Toggle data taking
        if key == ord('1'):
            takingData = not takingData

        elif key == ord('2'):
            showMask = not showMask

        # Toggle class
        elif ord('A') <= key <= ord('Z'):
            initClass(chr(key))
        elif ord('a') <= key <= ord('z'):
            initClass(chr(key).upper())

    cam.release()


if __name__ == '__main__':
    initClass(className)
    captureImages()

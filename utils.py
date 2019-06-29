#!/usr/bin/env python3
"""
Project Utilities.

Usage in python console: "from utils import function_name".

@author: Netanel Azoulay
@author: Roman Koifman
"""

import cv2
import glob
import os
import random
from projectParams import alphaBet


def flipImages(rootDir, imgFormat=None):
    """
    Flip all images in given rootDirectory. (Make new copies).

    :param rootDir: the folder that contain sub-folders which with images. Ex: "Images/train".
    :param imgFormat: 'jpg' or 'png'. If none provided, both will be used.
    """
    if imgFormat is None:
        flipImages(rootDir, 'jpg')
        flipImages(rootDir, 'png')
    else:
        string = rootDir + "/*/*." + imgFormat
        filenames = glob.glob(string)
        if len(filenames) == 0:
            string = rootDir + "/*." + imgFormat
            filenames = glob.glob(string)
        for fileName in filenames:
            img = cv2.imread(fileName)
            if img is not None:
                flippedFilename = fileName.replace("." + imgFormat, "_flipped." + imgFormat)
                cv2.imwrite(filename=flippedFilename, img=cv2.flip(img, 1))
                print("Flipped " + fileName + "as " + flippedFilename)


def binaryMask(img):
    """
    Apply binary mask on raw rgb image.

    :param img: 3D np array.
    :return: processed image. (3D np array).
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (7, 7), 3)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    ret, new = cv2.threshold(img, 25, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return new


def applyBinaryMasks(rootDir, imgFormat=None):
    """
    Apply binary mask on every img in the specified directory.

    :param rootDir: English path. must contain sub-folders which contains only images. Ex: "Images/train".
    :param imgFormat: 'jpg' or 'png'. If none provided, both will be used.
    """
    if imgFormat is None:
        applyBinaryMasks(rootDir, 'jpg')
        applyBinaryMasks(rootDir, 'png')
    else:
        string = rootDir + "/*/*." + imgFormat
        filenames = glob.glob(string)
        if len(filenames) == 0:
            string = rootDir + "/*." + imgFormat
            filenames = glob.glob(string)
        for fileName in filenames:
            img = cv2.imread(fileName)
            if img is not None:
                cv2.imwrite(filename=fileName, img=binaryMask(img))
                print("Applied Binary Mask on " + fileName)


def moveRandomFiles(from_dir, to_dir, percent):
    """
    Move random files between folders.

    :param from_dir: English path. Source directory.
    :param to_dir: English path. Destination directory.
    :param percent: Percent of files to move out of the source folder.
    """
    count = len(os.listdir(from_dir))
    numToMove = int(percent * count)
    try:
        os.makedirs(to_dir)
    except OSError as e:
        # print(e)
        pass

    for i in range(numToMove):  # [0,numToMove)
        fileName = random.choice(os.listdir(from_dir))
        os.rename(from_dir + "/" + fileName, to_dir + "/" + fileName)
        print("moved file " + fileName + " from " + from_dir + " to " + to_dir)


def moveProjectData(source="captureData/train", dest="captureData/test", percent=0.2):
    """
    Move random data between test and train folders.  iterate subdirectories.

    :param source: English path. Source directory.
    :param dest: English path. Destination directory.
    :param percent: Percent of files to move out of the source folder.
    :raises: OSError if source folder does not exist.
    """
    for subdir in os.listdir(source):
        moveRandomFiles(from_dir=source + "/" + subdir, to_dir=dest + "/" + subdir, percent=percent)


def convertIndexToHebrewLetter(index):
    """
    Convert index to hebrew letter.

    :param index: index in the range[0,23]. Out of range index will be converted to blank char.
    :return: Hebrew letter.
    """
    if index == 23:  # deletion
        return 'del'
    elif 0 <= index <= 22:  # 22 = space
        return alphaBet[index]
    else:
        return ''


def convertEnglishToHebrewLetter(englishLetter):
    """
    Convert english letter to hebrew letter.

    :param englishLetter: English letter.
    :return: Hebrew letter.
    """
    if englishLetter == ' ' or englishLetter == 'w' or englishLetter == 'W':
        return ' '
    elif englishLetter == 'x' or englishLetter == 'X':
        return 'del'
    elif 'a' <= englishLetter <= 'v':
        return convertIndexToHebrewLetter(ord(englishLetter) - ord('a'))
    elif 'A' <= englishLetter <= 'V':
        return convertIndexToHebrewLetter(ord(englishLetter) - ord('A'))
    else:
        return ''


def convertHebrewLetterToFinal(hebrewLetter):
    """
    Convert hebrew letter to final representation. Not will be changed if not convertable.

    :param hebrewLetter: Hebrew letter.
    :return: Final representation Hebrew letter.
    """
    if hebrewLetter == 'כ':
        return 'ך'
    elif hebrewLetter == 'מ':
        return 'ם'
    elif hebrewLetter == 'נ':
        return 'ן'
    elif hebrewLetter == 'פ':
        return 'ף'
    elif hebrewLetter == 'צ':
        return 'ץ'
    else:
        return hebrewLetter


def finalizeHebrewString(hebrewString):
    """
    Convert hebrew string letters to finals if needed (After space).

    :param hebrewString: Hebrew sentence.
    :return: Valid hebrew sentence with final letters representation.
    """
    if type("hebrewString") is not str or len(hebrewString) == 0:
        return hebrewString
    hebrewString = hebrewString.replace('כ ', 'ך ')
    hebrewString = hebrewString.replace('מ ', 'ם ')
    hebrewString = hebrewString.replace('נ ', 'ן ')
    hebrewString = hebrewString.replace('פ ', 'ף ')
    hebrewString = hebrewString.replace('צ ', 'ץ ')
    hebrewString = hebrewString[:-1] + convertHebrewLetterToFinal(hebrewString[-1])
    return hebrewString


def convertEnglishStringToHebrew(englishString):
    """
    Convert english string (representing ids) to hebrew string, finalizing final letters after space.

    :param englishString: english sentence.
    :return: Valid hebrew sentence with final letters representation.
    """
    for c in range(len(alphaBet)):
        eng1 = chr(ord('a') + c)
        eng2 = chr(ord('A') + c)
        englishString = englishString.replace(eng1, alphaBet[c])
        englishString = englishString.replace(eng2, alphaBet[c])
    return finalizeHebrewString(englishString)

import cv2
import glob
import os
import random
from projectParams import alphaBet


# Flip all images in given rootDirectory.
# Make a new copy.
# rootDir must contain sub-folders which contains only images. Ex: "Images/train".
def flipImages(rootDir, imgFormat=None):
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


# Apply binary mask on colored image.
# img = 3D np array.
def binaryMask(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (7, 7), 3)
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    ret, new = cv2.threshold(img, 25, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return new


# Apply binary mask on every img in the specified directory.
# English path only.
# rootDir must contain sub-folders which contains only images. Ex: "Images/train".
def applyBinaryMasks(rootDir, imgFormat=None):
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


# Move random files between folders.
# English path only.
def moveRandomFiles(from_dir, to_dir, percent):  # percent = % of files to take from the folder
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


# move random data between test and train folders.  iterate subdirectories.
# Folders must exist!!
# Ex for moving for validation data: "moveProjectData('images/train', 'images/validation', 0.1).
def moveProjectData(source="captureData/train", dest="captureData/test", percent=0.2):
    for subdir in os.listdir(source):
        moveRandomFiles(from_dir=source + "/" + subdir, to_dir=dest + "/" + subdir, percent=percent)


def convertIndexToHebrewLetter(index):
    if index == 23:  # deletion
        return 'del'
    elif 0 <= index <= 22:  # 22 = space
        return alphaBet[index]
    else:
        return ''


def convertEnglishToHebrewLetter(englishLetter):
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


# Convert hebrew string letters to finals if needed (After space).
def finalizeHebrewString(hebrewString):
    if type("hebrewString") is not str or len(hebrewString) == 0:
        return hebrewString
    hebrewString = hebrewString.replace('כ ', 'ך ')
    hebrewString = hebrewString.replace('מ ', 'ם ')
    hebrewString = hebrewString.replace('נ ', 'ן ')
    hebrewString = hebrewString.replace('פ ', 'ף ')
    hebrewString = hebrewString.replace('צ ', 'ץ ')
    hebrewString = hebrewString[:-1] + convertHebrewLetterToFinal(hebrewString[-1])
    return hebrewString


# Convert english string (representing keys) to hebrew string, finalizing final letters after space.
def convertEnglishStringToHebrew(englishString):
    for c in range(len(alphaBet)):
        eng1 = chr(ord('a') + c)
        eng2 = chr(ord('A') + c)
        englishString = englishString.replace(eng1, alphaBet[c])
        englishString = englishString.replace(eng2, alphaBet[c])
    return finalizeHebrewString(englishString)

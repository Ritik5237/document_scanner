# =====================================================
# Name: Ritik Sharma
# Roll No: 2301010342
# Course: Image Processing & Computer Vision
# Unit: Image Acquisition & Enhancement
# Assignment: Smart Document Scanner System
# Date: 09-Feb-2026
# =====================================================


import cv2
import os
import numpy as np


def welcome():
    print("="*60)
    print(" SMART DOCUMENT SCANNER SYSTEM ")
    print("="*60)
    print()


def load_image(path):

    img = cv2.imread(path)

    if img is None:
        return None, None

    img = cv2.resize(img, (512, 512))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img, gray


# Sampling
def sampling(gray):

    high = cv2.resize(gray, (512, 512))

    med = cv2.resize(gray, (256, 256))
    med = cv2.resize(med, (512, 512))

    low = cv2.resize(gray, (128, 128))
    low = cv2.resize(low, (512, 512))

    return high, med, low


def main():

    welcome()

    files = os.listdir("images")

    for file in files:

        path = "images/" + file

        img, gray = load_image(path)

        if img is None:
            continue

        high, med, low = sampling(gray)

        cv2.imshow("High", high)
        cv2.imshow("Medium", med)
        cv2.imshow("Low", low)

        cv2.waitKey(2000)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

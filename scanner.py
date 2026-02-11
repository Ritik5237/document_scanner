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


# Header
def welcome():
    print("="*60)
    print(" SMART DOCUMENT SCANNER SYSTEM ")
    print("="*60)
    print("Analyzing Sampling & Quantization Effects\n")


# Load Image
def load_image(path):

    img = cv2.imread(path)

    if img is None:
        print("Image not found:", path)
        return None, None

    img = cv2.resize(img, (512, 512))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    return img, gray


def main():

    welcome()

    folder = "images"
    files = os.listdir(folder)

    for file in files:

        path = os.path.join(folder, file)

        img, gray = load_image(path)

        if img is None:
            continue

        cv2.imshow("Original", img)
        cv2.imshow("Grayscale", gray)

        cv2.waitKey(2000)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

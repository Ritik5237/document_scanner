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


# -----------------------------------------------------
# Welcome Message
# -----------------------------------------------------
def welcome():

    print("=" * 60)
    print(" SMART DOCUMENT SCANNER SYSTEM ")
    print("=" * 60)
    print("Sampling & Quantization Analysis\n")


# -----------------------------------------------------
# Load and Preprocess Image
# -----------------------------------------------------
def load_image(path):

    image = cv2.imread(path)

    if image is None:
        print("Error: Cannot load image ->", path)
        return None, None

    # Resize to standard size
    image = cv2.resize(image, (512, 512))

    # Convert to Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return image, gray


# -----------------------------------------------------
# Sampling (Resolution Reduction)
# -----------------------------------------------------
def sampling(gray):

    # High Resolution (Original)
    high = cv2.resize(gray, (512, 512))

    # Medium Resolution
    medium = cv2.resize(gray, (256, 256))
    medium = cv2.resize(medium, (512, 512))

    # Low Resolution
    low = cv2.resize(gray, (128, 128))
    low = cv2.resize(low, (512, 512))

    return high, medium, low


# -----------------------------------------------------
# Quantization (Gray Level Reduction)
# -----------------------------------------------------
def quantize(image, levels):

    step = 256 // levels

    quantized = (image // step) * step

    return quantized.astype(np.uint8)


# -----------------------------------------------------
# Main Program
# -----------------------------------------------------
def main():

    welcome()

    image_folder = "images"

    # Check folder
    if not os.path.exists(image_folder):
        print("Images folder not found!")
        return

    files = os.listdir(image_folder)

    if len(files) == 0:
        print("No images found in images folder.")
        return

    for file in files:

        print("Processing:", file)

        path = os.path.join(image_folder, file)

        # Load image
        original, gray = load_image(path)

        if original is None:
            continue

        # ---------------- Sampling ----------------
        high, medium, low = sampling(gray)

        # ---------------- Quantization ----------------
        q8 = quantize(gray, 256)   # 8-bit
        q4 = quantize(gray, 16)    # 4-bit
        q2 = quantize(gray, 4)     # 2-bit

        # ---------------- Display ----------------

        cv2.imshow("Original", original)
        cv2.imshow("Grayscale", gray)

        cv2.imshow("High Resolution (512x512)", high)
        cv2.imshow("Medium Resolution (256x256)", medium)
        cv2.imshow("Low Resolution (128x128)", low)

        cv2.imshow("8-bit Quantization", q8)
        cv2.imshow("4-bit Quantization", q4)
        cv2.imshow("2-bit Quantization", q2)

        print("Showing results... Press any key")

        cv2.waitKey(0)
        cv2.destroyAllWindows()


# -----------------------------------------------------
# Program Entry
# -----------------------------------------------------
if __name__ == "__main__":
    main()

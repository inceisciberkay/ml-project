#!/usr/bin/python3

import cv2

def is_grayscale(image):
  """
  Checks if an image is grayscale.
  Args:
    image: The image to check.
  Returns:
    True if the image is grayscale, False otherwise.
  """
  # Convert the image to HSV color space.
  hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
  # Get the saturation channel of the image.
  saturation_channel = hsv_image[:, :, 1]
  # Check if all pixels in the saturation channel are 0.
  for pixel in saturation_channel:
    print(pixel)
    if pixel != 0:
      return False
  # All pixels in the saturation channel are 0, so the image is grayscale.
  return True


import numpy as np

img = cv2.imread('/home/inceisciberkay/cs464-project/project/all_images/0001_1297860395_01_WRI-L1_M014.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(img.shape)
print(np.mean(gray))
for i in range(727):
  for j in range(494):
    print(img[i,j,:])
    
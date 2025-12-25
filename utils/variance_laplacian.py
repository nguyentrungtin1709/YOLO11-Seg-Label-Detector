import cv2
import numpy as np

def laplacian_sharpness(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return lap.var()

img = cv2.imread("test.jpg")
score = laplacian_sharpness(img)

print("Sharpness score:", score)
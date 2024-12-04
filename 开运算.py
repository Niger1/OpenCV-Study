import cv2
import numpy as np

img = cv2.imread('dongwu.jpeg')
k = np.ones((10, 10), np.uint8)
r = cv2.morphologyEx(img, cv2.MORPH_OPEN, k)
cv2.imshow('morphologyEX', r)
cv2.waitKey()
cv2.destroyAllWindows()

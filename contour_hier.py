import cv2 as cv
import numpy as np

img = cv.imread('inputs/original.png', cv.IMREAD_UNCHANGED)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
edged = cv.Canny(gray, 127, 250)

contours, hier = cv.findContours(edged, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

print(hier)

# cv.imshow('Edged Image', edged)
# cv.waitKey(0)
# cv.destroyAllWindows()
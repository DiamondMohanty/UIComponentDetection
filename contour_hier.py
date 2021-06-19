import cv2 as cv
import numpy as np

# Blank Image
img = np.zeros((500,500,3), dtype='uint8')

# Drawing the outer box
rect_img = cv.rectangle(img, (125,125),(325, 325), (255,0,0), -1)

# Drawing the inner box
rect_img = cv.rectangle(rect_img, (200,200),(300, 300), (0,255,0), -1)

gray = cv.cvtColor(rect_img, cv.COLOR_RGB2GRAY)
edged = cv.Canny(gray, 127, 255)

# Finding the contours
contours, hier = cv.findContours(edged, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
print('Contours Found {0}'.format(len(contours)))

for comp in zip(contours, hier):
    print(comp[0])
    print(comp[1])

output = cv.drawContours(rect_img, contours, -1, (0,0,255))

cv.imshow('Image', rect_img)
cv.imshow('Contours', output)
cv.waitKey(0)
cv.destroyAllWindows()

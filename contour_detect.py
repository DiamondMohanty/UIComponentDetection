import cv2 as cv
import numpy as np


# reading image
img = cv.imread('inputs/original.png', cv.IMREAD_UNCHANGED)

# Gray Scale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Edged Image
edged = cv.Canny(gray, 127, 255)

# Countor Finding
contours, hier = cv.findContours(edged, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
print('Countours Found {0}'.format(len(contours)))

# Blank Image
blank = np.zeros((img.shape[0], img.shape[1], 3), dtype='uint8')
blank[:] = (255, 255, 255)

blank_tmp = np.zeros((img.shape[0], img.shape[1], 3), dtype='uint8')
blank_tmp[:] = (255, 255, 255)
blank_tmp = cv.drawContours(blank_tmp, contours, -1, (0,0,255))


output = img
for cnt in contours:
    # Calculate Contour Size
    x,y,w,h = cv.boundingRect(cnt)
    if h > 15 and w > 20:
        output = cv.rectangle(output, (x,y), (x+w, y+h), (255,0,0))        
        blank = cv.rectangle(blank, (x,y), (x+w, y+h), (255,0,0))        

# Display image
cv.imshow('Original Image', img)
cv.imshow('Edged Image', edged)
cv.imshow('Annotated Image', output)
cv.imshow('Blank Annotated Image', blank)
cv.imshow('Annotated Image', blank_tmp)
cv.waitKey(0)
cv.destroyAllWindows()